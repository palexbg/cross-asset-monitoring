from backend.ai.knowledge import METHODOLOGY_DEFINITIONS

#  Knowledge retrieval ---
METHODOLOGY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "lookup_methodology",
        "description": "Look up official definitions, methodology, logic, or how to interpret plots.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": list(METHODOLOGY_DEFINITIONS.keys())
                }
            },
            "required": ["topic"]
        }
    }
}

METHODOLOGY_BATCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "lookup_methodology_batch",
        "description": "Look up multiple official definitions/methodology blocks at once (useful when a question spans multiple concepts).",
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(METHODOLOGY_DEFINITIONS.keys())
                    },
                    "minItems": 1,
                    "maxItems": 8
                }
            },
            "required": ["topics"]
        }
    }
}


# --- Reporting ---
PORTFOLIO_STATUS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_portfolio_status",
        "description": "Get current NAV, return, and volatility.",
        "parameters": {"type": "object", "properties": {}}
    }
}


# aggregation and registry
def get_tool_schemas():
    """Returns the registry of all available tools."""
    return [
        METHODOLOGY_TOOL_SCHEMA,
        METHODOLOGY_BATCH_TOOL_SCHEMA,
        PORTFOLIO_STATUS_SCHEMA,
    ]
