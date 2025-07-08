# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

This is an **Intelligent Operations Agent** system that combines **LangGraph** for workflow orchestration and **DSPy** for modular reasoning. The system implements a three-layer architecture:

1. **LangGraph Workflow Layer**: State-driven orchestration (`src/langgraph_workflow/`)
2. **DSPy Reasoning Layer**: Modular AI components (`src/dspy_modules/`)
3. **Infrastructure Layer**: LLM integration and utilities (`src/utils/`, `src/agents/`)

### Core DSPy Components

The system is built around four main DSPy modules that implement Chain-of-Thought reasoning:

- **AlertAnalyzer**: Parses and classifies alerts with correlation analysis
- **DiagnosticAgent**: Performs root cause analysis and impact assessment
- **ActionPlanner**: Generates recovery strategies with risk assessment
- **ReportGenerator**: Creates incident reports and performance analysis

Each module extends `dspy.Module` and uses `dspy.ChainOfThought` for reasoning with structured signatures.

### LangGraph Workflow

The workflow follows a state-driven pattern using `StateGraph(OpsState)`:

```
monitor_collect → alert_process → fault_diagnose → action_plan → action_execute → report_generate → feedback_learn
```

All workflow nodes are in `src/langgraph_workflow/workflow_nodes.py` and use async/await patterns with comprehensive error handling.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies (requires uv)
uv sync                           # Production dependencies
uv sync --extra dev              # Include development tools
uv sync --extra docs             # Include documentation dependencies

# Virtual environment management
source .venv/bin/activate        # Activate virtual environment (macOS/Linux)
.venv\Scripts\activate           # Activate virtual environment (Windows)
deactivate                       # Exit virtual environment

# Add missing dependencies
uv add package-name              # Add production dependency
uv add --dev package-name        # Add development dependency
```

### Development Tools
```bash
# Activate virtual environment first
source .venv/bin/activate

# Code quality
uv run black .                   # Code formatting
uv run isort .                   # Import sorting
uv run flake8 .                  # Linting
uv run mypy src/                 # Type checking

# Testing
uv run pytest tests/unit/        # Unit tests
uv run pytest tests/integration/ # Integration tests
uv run pytest --cov=src         # Coverage testing

# Build
uv build                         # Build distribution packages
```

### Running Examples
```bash
# Activate virtual environment first
source .venv/bin/activate

# Basic workflow demonstration
python examples/basic_usage.py

# LLM configuration testing
python examples/deepseek_test.py

# Multi-agent collaboration
python examples/multi_agent_scenario.py

# Complete system demonstration
python examples/complete_demo.py
```

## LLM Configuration

The system uses a flexible LLM configuration system (`src/utils/llm_config.py`) that supports multiple providers:

### Primary: DeepSeek (Recommended)
```bash
# Environment variables
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export LLM_PROVIDER="deepseek"
export LLM_MODEL_NAME="deepseek-chat"
export LLM_BASE_URL="https://api.deepseek.com/v1"
```

### Alternative Providers
```bash
# OpenAI
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your-openai-api-key"

# Local Ollama
export LLM_PROVIDER="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"
```

## Key Implementation Patterns

### DSPy Module Structure
```python
class ModuleName(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reasoning_step = dspy.ChainOfThought(SignatureClass)
    
    def forward(self, input_data) -> OutputModel:
        # Implementation with structured reasoning
```

### LangGraph Node Pattern
```python
async def node_function(state: OpsState) -> OpsState:
    # Async processing with error handling
    try:
        # Integration with DSPy modules
        result = await some_dspy_module.process(state.data)
        return {"processed_data": result}
    except Exception as e:
        # Error handling and state update
        return {"error": str(e)}
```

### State Management
- Use `OpsState` (TypedDict) for all workflow state
- State includes: alerts, diagnostics, actions, reports, and metadata
- Access state history through `state.get("history", [])`

## Dependencies and Integration

### Core Dependencies
- `langgraph>=0.2.0`: Workflow orchestration
- `dspy-ai>=2.4.0`: Modular reasoning
- `langchain>=0.2.0`: LLM integration
- `langchain-openai`: OpenAI-compatible API support
- `pydantic>=2.0.0`: Data validation

### Development Dependencies
- `pytest>=7.0.0`: Testing framework with async support
- `black>=23.0.0`: Code formatting
- `mypy>=1.0.0`: Type checking
- `flake8>=6.0.0`: Code linting

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Test individual DSPy modules in isolation
- Mock LLM responses for consistent testing
- Focus on signature validation and output formatting

### Integration Tests (`tests/integration/`)
- Test complete workflows end-to-end
- Use real LLM providers (with API keys)
- Validate state transitions and error handling

### Running Specific Tests
```bash
# Single test file
uv run pytest tests/unit/test_alert_analyzer.py

# Specific test function
uv run pytest tests/unit/test_alert_analyzer.py::test_classify_alert

# With coverage
uv run pytest --cov=src/dspy_modules tests/unit/
```

## Troubleshooting

### Common Issues

1. **Virtual environment not activated**: Always run `source .venv/bin/activate` before executing Python commands
2. **Missing langchain_openai**: Run `uv add langchain-openai`
3. **LLM connection errors**: Verify API keys and network connectivity
4. **State validation errors**: Check `OpsState` structure in workflow nodes
5. **DSPy signature mismatch**: Ensure input/output models match module signatures

### Debugging Workflows
- Use `streaming=True` in workflow compilation for real-time monitoring
- Check state transitions with `state.get("history", [])`
- Enable debug logging: `LOG_LEVEL="DEBUG"` in environment

## Performance Considerations

- DSPy modules cache optimized prompts automatically
- LangGraph workflows support checkpointing for resumability
- Use async/await patterns throughout for concurrent processing
- Monitor token usage with DeepSeek for cost optimization