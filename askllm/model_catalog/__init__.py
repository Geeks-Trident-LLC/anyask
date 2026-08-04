from .model_registry import ModelRegistry


class model:
    class openai:
        default = ModelRegistry.default("openai")

    class azure:
        default = ModelRegistry.default("azure")

    class anthropic:
        default = ModelRegistry.default("anthropic")

    class gemini:
        default = ModelRegistry.default("gemini")

    class deepseek:
        default = ModelRegistry.default("deepseek")

    class groq:
        default = ModelRegistry.default("groq")

    class xai:
        default = ModelRegistry.default("xai")

    class together:
        default = ModelRegistry.default("together")

    class fireworks:
        default = ModelRegistry.default("fireworks")

    class cerebras:
        default = ModelRegistry.default("cerebras")

    class perplexity:
        default = ModelRegistry.default("perplexity")

    class openrouter:
        default = ModelRegistry.default("openrouter")

    class moonshot:
        default = ModelRegistry.default("moonshot")

    class mistral:
        default = ModelRegistry.default("mistral")

    class bedrock:
        default = ModelRegistry.default("bedrock")

    class cohere:
        default = ModelRegistry.default("cohere")

    class vertexai:
        default = ModelRegistry.default("vertexai")

    class oci:
        default = ModelRegistry.default("oci")
