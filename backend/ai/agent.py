import json
import os
import textwrap
from typing import Optional

from openai import OpenAI
from backend.ai.knowledge import get_definition
from backend.config import LLMConfig
from backend.ai.tools import get_tool_schemas


class PortfolioAgent:
    """A helpful AI agent for portfolio analysis tasks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        context: Optional[dict] = None
    ):

        self.config = config or LLMConfig()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable.")

        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.config.timeout_seconds
        )
        self.context = context or {}

        # Load Tool Definitions
        self.tools = get_tool_schemas()

        # Initialize System Prompt
        self.messages = []
        self._build_system_prompt()

    def update_context(self, new_context: dict):
        """
        Updates the agent's knowledge of the current dashboard state
        and rebuilds the system prompt to reflect the new data.
        """
        self.context = new_context
        self._build_system_prompt()
    # ----------------------------------

    def _answer_mentions_any_topic(self, answer: str, topics: list[str]) -> bool:
        if not answer:
            return False
        return any(t in answer for t in topics if t)

    def _rewrite_with_topic_citations(self, draft_answer: str, topics: list[str]) -> str:
        # Keep it short so you don't waste tokens on 4o-mini
        topics = [t for t in topics if t]
        cite_list = ", ".join(topics[:6])

        instruction = (
            "Rewrite your previous answer to be concise and grounded.\n"
            f"You MUST explicitly mention at least one methodology topic name from this list: {cite_list}.\n"
            "Do not invent topic names. Do not add new facts. Keep the same meaning."
        )

        rewrite = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=self.messages + [
                {"role": "assistant", "content": draft_answer},
                {"role": "user", "content": instruction},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens//2,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )
        return rewrite.choices[0].message.content

    # ----------------------------------
    # guard for long context

    def _format_context_value(self, v, max_chars: int = 4000) -> str:
        """
        Formats a context value for the LLM.
        if 'v' a markdown string, pass it through raw. Do not JSON dump it.
        """
        # if it is a string, use directly
        if isinstance(v, str):
            s = v
        # if dict/list/number, convert to string safely
        else:
            try:
                s = json.dumps(v, default=str)
            except Exception:
                s = str(v)

        # truncate if too long (keep in mind a table)
        if len(s) > max_chars:
            s = s[:max_chars] + " ...[truncated]"
        return s

    def _prune_messages(self, keep_last: int = 28):
        # keep system at index 0
        if not self.messages:
            return
        if self.messages[0].get("role") == "system":
            system = self.messages[0]
            rest = self.messages[1:]
            rest = rest[-keep_last:]
            self.messages = [system] + rest
        else:
            self.messages = self.messages[-(keep_last + 1):]

    # ----------------------------------

    def _build_system_prompt(self):
        """Constructs the system prompt string from the current context."""
        context_str = ""
        if self.context:
            context_lines = [
                f"- {k}: {self._format_context_value(v)}" for k, v in self.context.items()]

            context_str = "\nCURRENT SESSION CONTEXT:\n" + \
                "\n".join(context_lines)

        system_content = textwrap.dedent(f"""
            You are a Senior Financial Quantitative Risk Analyst.
            
            {context_str}

            RULES:
            1.1. **Methodology**: Use `lookup_methodology` for single definitions. If a question spans multiple concepts, call it multiple times or use `lookup_methodology_batch`. Synthesize the results. Quote the topic names.
            2. **Status**: **CHECK CONTEXT FIRST.** If the user asks for Volatility, Sharpe, or Drawdown, read 'summary_stats' from the Context above. ONLY call `get_portfolio_status` if the data is missing from Context.
            3. **Brevity**: Be concise. Do not ramble. When comparing risks, state the numbers clearly (e.g., "90% Market, 10% Specific").
            4. **Risk Views**: Total Risk can be decomposed in two separate ways by using Euler decomposition:
               - **By Asset**: How much assets contribute to total volatility. 
               - **By Factor**: How much each factor contributes. The remainder is "Idiosyncratic Risk".
               - **WARNING**: Do NOT equate "Asset Risk" with "Idiosyncratic Risk". They are completely different concepts.
        """).strip()

        # Update the system message (always index 0)
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = system_content
        else:
            self.messages.insert(
                0, {"role": "system", "content": system_content})

    def _lookup_methodology(self, topic: str) -> str:
        content = get_definition(topic)
        return json.dumps({"topic": topic, "content": content if content else "Topic not found."})

    def _lookup_methodology_batch(self, topics: list[str]) -> str:
        """Internal helper to fetch multiple definitions."""
        results = []
        for t in topics:
            content = get_definition(t)
            results.append({
                "topic": t,
                "content": content if content else "Topic not found."
            })
        return json.dumps({"results": results})

    def _get_portfolio_status(self) -> str:
        """Internal helper to fetch portfolio status from context."""
        if self.context and "summary_stats" in self.context:
            return json.dumps(self.context["summary_stats"])
        return json.dumps({"status": "error", "message": "No portfolio stats available in context."})

    def ask(self, user_input: str) -> str:
        """Main interaction loop."""
        self.messages.append({"role": "user", "content": user_input})
        self._prune_messages()

        # 1. First Pass: Think & Decide on Tools
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )

        msg = response.choices[0].message
        retrieved_topics = []

        # 2. Tool Execution Loop
        if msg.tool_calls:
            self.messages.append(msg)
            for tool in msg.tool_calls:
                tool_result = json.dumps({"error": "Tool execution failed."})
                try:
                    if tool.function.name == "lookup_methodology":
                        args = json.loads(tool.function.arguments)
                        t = args.get("topic")
                        if t:
                            retrieved_topics.append(t)
                        tool_result = self._lookup_methodology(t)
                    elif tool.function.name == "lookup_methodology_batch":
                        args = json.loads(tool.function.arguments)
                        ts = args.get("topics", [])
                        if isinstance(ts, list):
                            retrieved_topics.extend(
                                [x for x in ts if isinstance(x, str)])
                        tool_result = self._lookup_methodology_batch(ts)
                    elif tool.function.name == "get_portfolio_status":
                        tool_result = self._get_portfolio_status()
                    else:
                        tool_result = json.dumps(
                            {"error": f"Tool '{tool.function.name}' not found."})
                except Exception as e:
                    tool_result = json.dumps(
                        {"error": f"Internal error: {str(e)}"})

                self.messages.append({
                    "tool_call_id": tool.id,
                    "role": "tool",
                    "name": tool.function.name,
                    "content": tool_result
                })

            # 3. Second Pass: Synthesize Answer
            final = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=self.messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            answer = final.choices[0].message.content
        else:
            answer = msg.content

        # If we used methodology tools this turn, enforce topic-name citations via one rewrite
        if retrieved_topics and not self._answer_mentions_any_topic(answer, retrieved_topics):
            answer = self._rewrite_with_topic_citations(
                answer, retrieved_topics)
        self.messages.append({"role": "assistant", "content": answer})

        # prune message (FIFO) so that we do not lose the instructions if the chat becomes too big.
        self._prune_messages()

        return answer
