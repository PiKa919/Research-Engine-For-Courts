# Contributing to Legal Research Engine

First off, thank you for considering contributing to the Legal Research Engine! 🎉

It's people like you that make this project truly beneficial for the legal community worldwide.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [legal-research-engine@yourdomain.com].

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/your-username/legal-research-engine/issues) as you might find out that you don't need to create one.

When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps which reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include screenshots and animated GIFs if possible**

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/your-username/legal-research-engine/issues). When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain which behavior you expected to see instead**
- **Explain why this enhancement would be useful to most Legal Research Engine users**

### 🔧 Code Contributions

Unsure where to begin contributing? You can start by looking through these `beginner` and `help-wanted` issues:

- [Beginner issues](https://github.com/your-username/legal-research-engine/issues?q=is%3Aissue+is%3Aopen+label%3Abeginner) - issues which should only require a few lines of code
- [Help wanted issues](https://github.com/your-username/legal-research-engine/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) - issues which should be a bit more involved

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- LM Studio (for local testing)

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/legal-research-engine.git
   cd legal-research-engine
   ```

3. Create a virtual environment:
   ```bash
   python -m venv legal_env
   source legal_env/bin/activate  # On Windows: legal_env\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

6. Create a branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

### Project Structure

Understanding the project structure will help you navigate and contribute effectively:

```
legal-research-engine/
├── src/                          # Source code
│   ├── langgraph_workflow.py     # Multi-agent workflows
│   ├── langgraph_integration.py  # Integration layer
│   ├── streamlit_enhancements.py # UI components
│   ├── rag_pipeline.py          # RAG implementation
│   └── ...
├── tests/                        # Test suite
├── docs/                         # Documentation
├── data/                         # Sample legal documents
├── app.py                        # Main application
└── requirements*.txt             # Dependencies
```

### Local Development

1. **Start LM Studio**: Ensure LM Studio is running locally on port 1234
2. **Run tests**: `python -m pytest tests/`
3. **Start the app**: `streamlit run app.py`
4. **Make your changes**: Edit the relevant files
5. **Test your changes**: Run both automated tests and manual testing

### Adding New Features

When adding new features:

1. **Create tests first** (Test-Driven Development)
2. **Update documentation** as you go
3. **Follow existing patterns** in the codebase
4. **Add type hints** where appropriate
5. **Include error handling**

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some additional conventions:

- **Line length**: 88 characters (Black formatter default)
- **Imports**: Use absolute imports, group them properly
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public functions/classes

### Code Formatting

We use several tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

Examples:
```
feat(langgraph): add precedent analysis agent
fix(ui): resolve chat interface scrolling issue
docs: update API documentation for new endpoints
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_langgraph_workflow.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names: `test_should_extract_citations_from_legal_document`
- Follow the AAA pattern: Arrange, Act, Assert

Example test:
```python
def test_should_generate_case_brief_with_valid_input():
    # Arrange
    generator = CaseBriefGenerator()
    document = Document(page_content="Sample legal content", metadata={})
    
    # Act
    result = generator.generate_case_brief(document)
    
    # Assert
    assert result.case_name is not None
    assert len(result.legal_issues) > 0
```

### Test Categories

- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete user workflows
- **Performance Tests**: Ensure acceptable response times

## Pull Request Process

1. **Ensure your PR addresses an existing issue** (create one if needed)
2. **Update the README.md** with details of changes if applicable
3. **Add tests** for your changes
4. **Ensure all tests pass** locally
5. **Update documentation** if you've changed APIs
6. **Follow the PR template** when creating your pull request

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Testing** in different environments if needed
4. **Documentation review** for user-facing changes

## Community

### Communication Channels

- **GitHub Discussions**: For questions and general discussions
- **GitHub Issues**: For bugs and feature requests
- **Email**: For private communications
- **Discord/Slack**: Real-time community chat (coming soon)

### Recognition

Contributors are recognized in several ways:

- **Contributors list** in README
- **Release notes** mention for significant contributions
- **Special badges** for regular contributors
- **Conference speaking opportunities** for major contributors

## Development Guidelines

### LangGraph Workflows

When working with LangGraph workflows:

- **State management**: Ensure proper state transitions
- **Error handling**: Implement retry logic and graceful failures
- **Monitoring**: Add appropriate logging and metrics
- **Testing**: Mock external dependencies appropriately

### Streamlit Components

For Streamlit UI development:

- **Session state**: Use session state appropriately
- **Caching**: Leverage Streamlit caching for performance
- **Responsive design**: Ensure mobile compatibility
- **Accessibility**: Follow web accessibility guidelines

### Legal Domain Knowledge

- **Accuracy**: Ensure legal accuracy in features
- **Privacy**: Maintain user data privacy
- **Compliance**: Consider legal compliance requirements
- **Internationalization**: Consider different legal systems

## Getting Help

Don't hesitate to ask for help:

1. **Check existing documentation** first
2. **Search closed issues** for similar problems
3. **Ask in GitHub Discussions** for general questions
4. **Create a draft PR** for code-related questions
5. **Contact maintainers** directly for urgent matters

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Legal Research Engine! 🏛️✨