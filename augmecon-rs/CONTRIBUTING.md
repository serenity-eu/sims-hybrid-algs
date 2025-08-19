# Contributing to AUGMECON-RS

Thank you for your interest in contributing to AUGMECON-RS! This guide will help you get started with contributing to our multi-objective optimization library.

## 🎯 Ways to Contribute

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new functionality or improvements
- **Code Contributions**: Submit patches, new features, or optimizations
- **Documentation**: Improve docs, add examples, write tutorials
- **Testing**: Add test cases, improve test coverage, performance testing
- **Community**: Help answer questions, review PRs, participate in discussions

## 🚀 Getting Started

### Prerequisites

- Rust 1.70 or later
- Git
- Basic understanding of optimization and/or Rust

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/augmecon-rs.git
   cd augmecon-rs
   ```

2. **Install Dependencies**
   ```bash
   # Rust toolchain (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Development tools
   cargo install cargo-fmt cargo-clippy
   ```

3. **Build and Test**
   ```bash
   # Build the project
   cargo build
   
   # Run tests
   cargo test
   
   # Run examples
   cargo run --example basic_production_simple
   ```

4. **Verify Everything Works**
   ```bash
   # Run all checks that CI will run
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test --all-features
   cargo doc --no-deps
   ```

## 📝 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write clean, documented code
- Follow Rust conventions and idioms
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run the full test suite
cargo test

# Test specific modules
cargo test test_two_objectives

# Run with logging to debug
RUST_LOG=debug cargo test -- --nocapture

# Test examples
cargo run --example basic_production_simple
```

### 4. Format and Lint

```bash
# Format code
cargo fmt

# Run clippy for linting
cargo clippy -- -D warnings

# Check documentation
cargo doc --no-deps --open
```

### 5. Commit and Push

```bash
git add .
git commit -m "feat: add new AUGMECON variant implementation"
git push origin feature/your-feature-name
```

### 6. Submit a Pull Request

- Use a clear, descriptive title
- Describe what your changes do and why
- Reference any related issues
- Include screenshots for UI changes
- Ensure all CI checks pass

## 🧪 Testing Guidelines

### Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_specific_functionality() {
        // Arrange
        let problem = create_test_problem();
        
        // Act
        let result = solve_problem(problem);
        
        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), expected_count);
    }
}
```

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test module interactions
3. **Example Tests**: Ensure examples compile and run
4. **Performance Tests**: Benchmark critical paths
5. **Regression Tests**: Prevent known issues from returning

### Adding Test Cases

```rust
// tests/test_new_feature.rs
use augmecon::*;

#[test]
fn test_new_feature_basic_case() {
    let mut problem = MultiObjectiveProblem::new();
    // Set up test problem...
    
    let options = Options::new().with_grid_points(10);
    let mut solver = Augmecon::new(problem, options).unwrap();
    
    assert!(solver.solve().is_ok());
    assert!(!solver.get_pareto_solutions().is_empty());
}
```

## 📚 Documentation Standards

### Code Documentation

```rust
/// Brief one-line description of the function.
///
/// More detailed description if needed. Explain what the function does,
/// when to use it, and any important considerations.
///
/// # Arguments
///
/// * `param1` - Description of the first parameter
/// * `param2` - Description of the second parameter
///
/// # Returns
///
/// Description of what the function returns
///
/// # Errors
///
/// Describe when and why this function might return an error
///
/// # Examples
///
/// ```rust
/// use augmecon::*;
/// 
/// let result = your_function(param1, param2)?;
/// assert_eq!(result, expected_value);
/// ```
///
/// # Panics
///
/// Describe any conditions that cause panics (avoid these if possible)
pub fn your_function(param1: Type1, param2: Type2) -> Result<ReturnType> {
    // Implementation
}
```

### Module Documentation

```rust
//! # Module Name
//!
//! Brief description of what this module does.
//!
//! ## Overview
//!
//! Longer description with key concepts and usage patterns.
//!
//! ## Examples
//!
//! ```rust
//! use augmecon::module_name::*;
//! 
//! // Example usage
//! ```
```

## 🎨 Code Style

### Formatting

We use `rustfmt` with default settings:

```bash
cargo fmt
```

### Naming Conventions

- **Types**: `PascalCase` (e.g., `MultiObjectiveProblem`)
- **Functions/Variables**: `snake_case` (e.g., `solve_problem`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_GRID_POINTS`)
- **Modules**: `snake_case` (e.g., `solution_analysis`)

### Error Handling

```rust
// Prefer Result types over panics
fn fallible_operation() -> Result<T, AugmeconError> {
    // Implementation
}

// Use thiserror for custom errors
#[derive(Error, Debug)]
pub enum CustomError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}
```

### Performance Considerations

```rust
// Prefer borrowing over cloning
fn process_solutions(solutions: &[Solution]) -> Vec<f64> {
    solutions.iter().map(|s| s.compute_metric()).collect()
}

// Use appropriate data structures
use std::collections::HashMap; // For key-value lookups
use std::collections::BTreeMap; // For ordered data
use Vec; // For sequential data
```

## 🐛 Bug Reports

### Before Reporting

1. Check if the issue already exists
2. Try to reproduce with minimal example
3. Test with the latest version

### Report Template

```markdown
**Bug Description**
Clear description of what went wrong.

**To Reproduce**
Steps or code to reproduce the issue:
```rust
// Minimal reproducible example
```

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Linux, Windows, macOS]
- Rust version: [e.g., 1.70.0]
- AUGMECON-RS version: [e.g., 0.1.0]

**Additional Context**
Any other relevant information.
```

## ✨ Feature Requests

### Request Template

```markdown
**Feature Description**
Clear description of the proposed feature.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed API**
```rust
// Example of how the feature might be used
```

**Alternatives Considered**
Other approaches you've considered.

**Implementation Notes**
Any ideas about implementation approach.
```

## 🔍 Code Review Process

### For Contributors

- **Small PRs**: Focus on single features/fixes
- **Clear Description**: Explain what and why
- **Test Coverage**: Include relevant tests
- **Documentation**: Update docs for public APIs
- **Performance**: Consider performance implications

### For Reviewers

- **Be Constructive**: Provide helpful feedback
- **Test Locally**: Verify changes work as expected
- **Check Standards**: Ensure code follows project conventions
- **Security**: Look for potential security issues
- **Performance**: Consider algorithmic complexity

### Review Checklist

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] New functionality is tested
- [ ] Documentation is updated
- [ ] Performance is acceptable
- [ ] Error handling is appropriate
- [ ] Code follows style guidelines

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] CHANGELOG is updated
- [ ] Version numbers are bumped
- [ ] Release notes are prepared
- [ ] Crates.io publication is ready

## 🤝 Community Guidelines

### Code of Conduct

- **Be Respectful**: Treat everyone with respect
- **Be Inclusive**: Welcome people of all backgrounds
- **Be Patient**: Help newcomers learn
- **Be Professional**: Maintain a professional tone

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions and reviews

### Getting Help

- **Documentation**: Check the docs first
- **Examples**: Look at existing examples
- **Issues**: Search existing issues
- **Discussions**: Ask questions in discussions

## 📋 Contributor Checklist

Before submitting your first contribution:

- [ ] Read this contributing guide
- [ ] Set up development environment
- [ ] Run tests successfully
- [ ] Understand code style guidelines
- [ ] Know how to submit PRs

For each contribution:

- [ ] Create appropriate branch
- [ ] Write/update tests
- [ ] Update documentation
- [ ] Follow code style
- [ ] Write clear commit messages
- [ ] Submit detailed PR

## 🙏 Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **Documentation**: Attribution where appropriate

Thank you for contributing to AUGMECON-RS! Your contributions help make multi-objective optimization more accessible to the Rust community.

## 📞 Questions?

If you have any questions about contributing, please:

1. Check existing documentation
2. Search GitHub issues and discussions
3. Open a new discussion
4. Tag maintainers if urgent

We're here to help and appreciate your interest in improving AUGMECON-RS!
