# Contributing to UPIR

Thank you for your interest in contributing to UPIR!

## How to Contribute

This is an internal Google experimental research project. Contributions are welcome from Googlers.

### Before Contributing

1. Familiarize yourself with the [Google Open Source documentation](http://go/releasing)
2. Review the Google Python Style Guide at http://go/py-style
3. Ensure your changes align with the project's experimental nature

### Development Process

1. **Create a branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow Google's Python style guide
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   python -m pytest tests/
   ```

4. **Submit for review**
   - Create a pull request
   - Add relevant reviewers
   - Include a clear description of changes

### Code Style

- Python code should follow [Google Python Style Guide](http://go/py-style)
- Use type hints where appropriate
- Maximum line length: 80 characters
- Use descriptive variable names

### Testing

- All new features must include tests
- Maintain or improve code coverage
- Tests should be in the `tests/` directory

### Documentation

- Update README.md if adding new features
- Add docstrings to all public functions
- Include usage examples where appropriate

## Questions?

Contact: subhadipmitra@google.com

## Note

This is an experimental research project and not an official Google product.