# Claude GeoAI Skills

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Skills](https://img.shields.io/badge/Skills-52-brightgreen.svg)](#whats-included)

A comprehensive collection of **52 ready-to-use skills** for Claude, focused on **Geospatial AI (GeoAI)** and scientific computing. Transform Claude into your AI research assistant capable of executing complex multi-step workflows involving spatial data analysis, machine learning, visualization, and scientific communication.

These skills enable Claude to seamlessly work with specialized libraries, databases, and tools across multiple domains:
- 🌍 Geospatial Analysis & Visualization - Spatial data processing, mapping, GIS operations
- 🤖 Machine Learning & Deep Learning - Classical ML, deep learning, time series, graph ML, Bayesian methods
- 📊 Data Analysis & Big Data - Statistical analysis, large-scale data processing, exploratory analysis
- 📈 Visualization & Scientific Graphics - Publication-quality plots, interactive visualizations, schematics
- 📚 Research & Scientific Communication - Literature search, scientific writing, peer review, presentations
- 🔧 Infrastructure & Cloud Computing - Serverless platforms, resource detection, workflow automation

**Transform Claude Code into an 'AI GeoScientist' on your desktop!**

> ⭐ **If you find this repository useful**, please consider giving it a star! It helps others discover these tools and encourages us to continue maintaining and expanding this collection.

---

## 📦 What's Included

This repository provides **52 skills** organized into the following categories:

- **3+ Scientific Databases** - Direct API access to OpenAlex, PubMed, and Data Commons
- **25+ Python Packages** - Geopandas, PyTorch Lightning, scikit-learn, Transformers, and more
- **20+ Analysis & Communication Tools** - Literature review, scientific writing, visualization, document processing, and more
- **4+ Research Tools** - Hypothesis generation, grant writing, critical thinking, and research evaluation

Each skill includes:
- ✅ Comprehensive documentation (`SKILL.md`)
- ✅ Practical code examples
- ✅ Use cases and best practices
- ✅ Integration guides
- ✅ Reference materials

---

## 📋 Table of Contents

- [What's Included](#whats-included)
- [Why Use This?](#why-use-this)
- [Getting Started](#getting-started)
  - [Claude Code](#claude-code-recommended)
- [Prerequisites](#prerequisites)
- [Quick Examples](#quick-examples)
- [Use Cases](#use-cases)
- [Available Skills](#available-skills)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Support](#support)
- [Citation](#citation)
- [License](#license)

---

## 🚀 Why Use This?

### ⚡ **Accelerate Your GeoAI Research**
- **Save Days of Work** - Skip API documentation research and integration setup
- **Production-Ready Code** - Tested, validated examples following scientific best practices
- **Multi-Step Workflows** - Execute complex pipelines with a single prompt

### 🎯 **GeoAI-Focused Coverage**
- **52 Skills** - Extensive coverage across geospatial, ML, and scientific domains
- **3+ Databases** - Direct access to OpenAlex, PubMed, and Data Commons
- **25+ Python Packages** - Geopandas, PyTorch Lightning, scikit-learn, Transformers, and others

### 🔧 **Easy Integration**
- **Simple Installation** - Install via Claude Code marketplace
- **Automatic Discovery** - Claude automatically finds and uses relevant skills
- **Well Documented** - Each skill includes examples, use cases, and best practices

### 🌟 **Maintained & Supported**
- **Regular Updates** - Continuously maintained and expanded
- **Community Driven** - Open source with active community contributions
- **Enterprise Ready** - Commercial support available for advanced needs

---

## 🎯 Getting Started

Choose your preferred platform to get started:

### 🖥️ Claude Code (Recommended)

> 📚 **New to Claude Code?** Check out the [Claude Code Quickstart Guide](https://docs.claude.com/en/docs/claude-code/quickstart) to get started.

**Step 1: Install Claude Code**

**macOS:**
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows:**
```powershell
irm https://claude.ai/install.ps1 | iex
```

**Step 2: Register the Marketplace**

```bash
/plugin marketplace add sacrrie/claude-geoai-skills
```

**Step 3: Install Skills**

1. Open Claude Code
2. Select **Browse and install plugins**
3. Choose **claude-geoai-skills**
4. Select **geoai-skills**
5. Click **Install now**

**That's it!** Claude will automatically use the appropriate skills when you describe your geospatial and scientific tasks. Make sure to keep the skill up to date!

---

## ⚙️ Prerequisites

- **Python**: 3.9+ (3.12+ recommended for best compatibility)
- **uv**: Python package manager (required for installing skill dependencies)
- **Client**: Claude Code
- **System**: macOS, Linux, or Windows with WSL2
- **Dependencies**: Automatically handled by individual skills (check `SKILL.md` files for specific requirements)

### Installing uv

The skills use `uv` as the package manager for installing Python dependencies. Install it using the instructions for your operating system:

**macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (via pip):**
```bash
pip install uv
```

After installation, verify it works by running:
```bash
uv --version
```

For more installation options and details, visit the [official uv documentation](https://docs.astral.sh/uv/).

---

## 💡 Quick Examples

Once you've installed the skills, you can ask Claude to execute complex multi-step GeoAI workflows. Here are some example prompts:

### 🌍 Geospatial Data Analysis Pipeline
**Goal**: Analyze urban spatial patterns and visualize results

**Prompt**:
```
Use available skills you have access to whenever possible. Load shapefile data with GeoPandas, perform spatial joins and buffering operations, extract statistical features with pandas, visualize spatial patterns with Plotly interactive maps, create publication-quality figures with matplotlib and seaborn, and generate a comprehensive analysis report.
```

**Skills Used**: GeoPandas, Plotly, Matplotlib, Seaborn, Scientific Visualization

---

### 🤖 Machine Learning for Spatial Prediction
**Goal**: Build a predictive model for spatial phenomena

**Prompt**:
```
Use available skills you have access to whenever possible. Load spatial training data with GeoPandas, preprocess features with scikit-learn, train a deep learning model with PyTorch Lightning, evaluate model performance with appropriate metrics, interpret feature importance with SHAP, visualize predictions with Plotly, and document methodology with Scientific Writing skill.
```

**Skills Used**: GeoPandas, scikit-learn, PyTorch Lightning, SHAP, Plotly, Scientific Writing

---

### 📊 Large-Scale Spatial Data Processing
**Goal**: Process massive geospatial datasets efficiently

**Prompt**:
```
Use available skills you have access to whenever possible. Detect available computational resources, process large geospatial datasets with Dask for parallel computation, handle data with Polars for performance, store results in Zarr format for efficient chunked access, and generate analysis reports with LaTeX.
```

**Skills Used**: Get Available Resources, Dask, Polars, Zarr, LaTeX Posters

---

### 📚 Literature Review & Research Synthesis
**Goal**: Comprehensive survey of GeoAI techniques

**Prompt**:
```
Use available skills you have access to whenever possible. Search PubMed and OpenAlex for recent GeoAI literature, perform systematic literature review with proper citation management, synthesize findings with Peer Review criteria, generate visualizations with Plotly and matplotlib, and create publication-ready slides with Scientific Slides skill.
```

**Skills Used**: PubMed, OpenAlex, Literature Review, Citation Management, Peer Review, Plotly, Matplotlib, Scientific Slides

---

### 🎯 Interactive Geospatial Dashboard
**Goal**: Create an interactive dashboard for spatial data exploration

**Prompt**:
```
Use available skills you have access to whenever possible. Load spatial data with GeoPandas, perform exploratory data analysis, create interactive visualizations with Plotly including maps and time series, generate publication-quality static plots with matplotlib/seaborn, and create professional slides for presentation.
```

**Skills Used**: GeoPandas, Exploratory Data Analysis, Plotly, Matplotlib, Seaborn, Scientific Slides

---

## 🔬 Use Cases

### 🌍 Geospatial Analysis & Visualization
- **Spatial Data Processing**: Load, manipulate, and analyze geospatial vector data with GeoPandas
- **Interactive Mapping**: Create interactive web-based maps and visualizations with Plotly
- **Statistical Analysis**: Perform spatial statistics and hypothesis testing with scikit-learn and statsmodels
- **Publication Figures**: Create publication-quality maps and plots with matplotlib and seaborn

### 🤖 Machine Learning & AI
- **Classical ML**: Train and evaluate traditional ML models with scikit-learn
- **Deep Learning**: Build and train neural networks with PyTorch Lightning
- **Model Interpretability**: Explain model predictions with SHAP
- **Graph ML**: Work with graph neural networks using Torch Geometric
- **Time Series Analysis**: Analyze temporal data with aeon

### 📊 Data Analysis & Big Data
- **Large-Scale Processing**: Handle datasets larger than RAM with Dask and Vaex
- **High-Performance Computing**: Use Polars for fast data manipulation
- **Exploratory Analysis**: Automated EDA with statistics and visualizations
- **Statistical Modeling**: Perform Bayesian analysis with PyMC, survival analysis with scikit-survival

### 📈 Visualization & Scientific Communication
- **Interactive Visualizations**: Create dashboards with Plotly
- **Publication Graphics**: Generate high-quality figures with matplotlib and seaborn
- **Scientific Schematics**: Create diagrams with Generate Image skill
- **Presentations**: Build slides and posters with Scientific Slides and LaTeX Posters

### 📚 Research & Scientific Communication
- **Literature Review**: Systematic searches across PubMed and OpenAlex
- **Scientific Writing**: Write research papers with proper structure and citations
- **Peer Review**: Conduct rigorous peer reviews
- **Grant Writing**: Prepare competitive research proposals

---

## 📚 Available Skills

This repository contains **52 skills** organized across multiple domains. Each skill provides comprehensive documentation, code examples, and best practices for working with scientific libraries, databases, and tools.

### Skill Categories

#### 🤖 **Machine Learning & Deep Learning** (12+ skills)
- Deep learning: PyTorch Lightning, Transformers, Stable Baselines3, PufferLib
- Classical ML: scikit-learn, scikit-survival, SHAP
- Time series: aeon
- Bayesian methods: PyMC
- Optimization: PyMOO
- Graph ML: Torch Geometric
- Dimensionality reduction: UMAP-learn
- Statistical modeling: statsmodels

#### 📊 **Data Analysis & Visualization** (14+ skills)
- Visualization: Matplotlib, Seaborn, Plotly, Scientific Visualization
- Geospatial analysis: GeoPandas
- Network analysis: NetworkX
- Symbolic math: SymPy
- Data access: Data Commons
- Exploratory data analysis: EDA workflows
- Statistical analysis: Statistical Analysis workflows

#### 🚀 **Infrastructure & Cloud Computing** (4+ skills)
- Cloud compute: Modal
- Tool discovery: Get Available Resources
- High-performance data: Dask, Polars, Vaex, Zarr

#### 📚 **Scientific Communication** (12+ skills)
- Literature: OpenAlex, PubMed, Literature Review
- Web search: Perplexity Search (AI-powered search with real-time information)
- Writing: Scientific Writing, Peer Review
- Document processing: MarkItDown, Document Skills
- Publishing: Paper-2-Web, Venue Templates
- Presentations: Scientific Slides, LaTeX Posters, PPTX Posters
- Diagrams: Scientific Schematics
- Citations: Citation Management
- Illustration: Generate Image (AI image generation)

#### 🔬 **Scientific Databases** (3+ skills)
- Literature: OpenAlex, PubMed
- Public data: Data Commons

#### 🎓 **Research Methodology & Planning** (7+ skills)
- Ideation: Scientific Brainstorming, Hypothesis Generation
- Critical analysis: Scientific Critical Thinking, Scholar Evaluation
- Funding: Research Grants
- Discovery: Research Lookup
- Market analysis: Market Research Reports

> 📖 **For complete details on all skills**, see [docs/scientific-skills.md](docs/scientific-skills.md)

> 💡 **Looking for practical examples?** Check out [docs/examples.md](docs/examples.md) for comprehensive workflow examples across all scientific domains.

---

## 🤝 Contributing

We welcome contributions to expand and improve this GeoAI skills repository!

### Ways to Contribute

✨ **Add New Skills**
- Create skills for additional geospatial or scientific computing packages
- Add integrations for geospatial platforms and tools

📚 **Improve Existing Skills**
- Enhance documentation with more examples and use cases
- Add new workflows and reference materials
- Improve code examples and scripts
- Fix bugs or update outdated information

🐛 **Report Issues**
- Submit bug reports with detailed reproduction steps
- Suggest improvements or new features

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-skill`)
3. **Follow** the existing directory structure and documentation patterns
4. **Ensure** all new skills include comprehensive `SKILL.md` files
5. **Test** your examples and workflows thoroughly
6. **Commit** your changes (`git commit -m 'Add amazing skill'`)
7. **Push** to your branch (`git push origin feature/amazing-skill`)
8. **Submit** a pull request with a clear description of your changes

### Contribution Guidelines

✅ Maintain consistency with existing skill documentation format
✅ Include practical, working examples in all contributions
✅ Ensure all code examples are tested and functional
✅ Follow scientific best practices in examples and workflows
✅ Update relevant documentation when adding new capabilities
✅ Provide clear comments and docstrings in code
✅ Include references to official documentation

### Recognition

Contributors are recognized in our community and may be featured in:
- Repository contributors list
- Special mentions in release notes
- Community highlights

Your contributions help make GeoAI computing more accessible and enable researchers to leverage AI tools more effectively!

---

## 🔧 Troubleshooting

### Common Issues

**Problem: Skills not loading in Claude Code**
- Solution: Ensure you've installed the latest version of Claude Code
- Try reinstalling the plugin: `/plugin marketplace add sacrrie/claude-geoai-skills`

**Problem: Missing Python dependencies**
- Solution: Check the specific `SKILL.md` file for required packages
- Install dependencies: `uv pip install package-name`

**Problem: API rate limits**
- Solution: Many databases have rate limits. Review the specific database documentation
- Consider implementing caching or batch requests

**Problem: Authentication errors**
- Solution: Some services require API keys. Check the `SKILL.md` for authentication setup
- Verify your credentials and permissions

**Problem: Outdated examples**
- Solution: Report the issue via GitHub Issues
- Check the official package documentation for updated syntax

---

## ❓ FAQ

### General Questions

**Q: Is this free to use?**
A: Yes! This repository is MIT licensed. However, each individual skill has its own license specified in the `license` metadata field within its `SKILL.md` file—be sure to review and comply with those terms.

**Q: Can I use this for commercial projects?**
A: The repository itself is MIT licensed, which allows commercial use. However, individual skills may have different licenses—check the `license` field in each skill's `SKILL.md` file to ensure compliance with your intended use.

**Q: Do all skills have the same license?**
A: No. Each skill has its own license specified in the `license` metadata field within its `SKILL.md` file. These licenses may differ from the repository's MIT License. Users are responsible for reviewing and adhering to the license terms of each individual skill they use.

**Q: How often is this updated?**
A: We regularly update skills to reflect the latest versions of packages and APIs. Major updates are announced in release notes.

**Q: Can I use this with other AI models?**
A: The skills are optimized for Claude Code.

### Installation & Setup

**Q: Do I need all the Python packages installed?**
A: No! Only install the packages you need. Each skill specifies its requirements in its `SKILL.md` file.

**Q: What if a skill doesn't work?**
A: First check the [Troubleshooting](#troubleshooting) section. If the issue persists, file an issue on GitHub with detailed reproduction steps.

**Q: Do the skills work offline?**
A: Database skills require internet access to query APIs. Package skills work offline once Python dependencies are installed.

### Contributing

**Q: Can I contribute my own skills?**
A: Absolutely! We welcome contributions. See the [Contributing](#contributing) section for guidelines and best practices.

**Q: How do I report bugs or suggest features?**
A: Open an issue on GitHub with a clear description. For bugs, include reproduction steps and expected vs actual behavior.

---

## 💬 Support

Need help? Here's how to get support:

- 📖 **Documentation**: Check the relevant `SKILL.md` and `references/` folders
- 🐛 **Bug Reports**: [Open an issue](https://github.com/sacrrie/claude-geoai-skills/issues)
- 💡 **Feature Requests**: [Submit a feature request](https://github.com/sacrrie/claude-geoai-skills/issues/new)

---

## 📖 Citation

If you use Claude GeoAI Skills in your research or project, please cite it as:

### BibTeX
```bibtex
@software{claude_geoai_skills_2025,
  author = {{sacrrie}},
  title = {Claude GeoAI Skills: A Comprehensive Collection of Geospatial AI Tools for Claude AI},
  year = {2025},
  url = {https://github.com/sacrrie/claude-geoai-skills},
  note = {skills covering geospatial analysis, machine learning, visualization, and scientific communication}
}
```

### APA
```
sacrrie. (2025). Claude GeoAI Skills: A comprehensive collection of geospatial AI tools for Claude AI [Computer software]. https://github.com/sacrrie/claude-geoai-skills
```

### MLA
```
sacrrie. Claude GeoAI Skills: A Comprehensive Collection of Geospatial AI Tools for Claude AI. 2025, github.com/sacrrie/claude-geoai-skills.
```

### Plain Text
```
Claude GeoAI Skills by sacrrie (2025)
Available at: https://github.com/sacrrie/claude-geoai-skills
```

We appreciate acknowledgment in publications, presentations, or projects that benefit from these skills!

---

## 📄 License

This project is licensed under the **MIT License**.

**Copyright © 2025 sacrrie**

### Key Points:
- ✅ **Free for any use** (commercial and noncommercial)
- ✅ **Open source** - modify, distribute, and use freely
- ✅ **Permissive** - minimal restrictions on reuse
- ⚠️ **No warranty** - provided "as is" without warranty of any kind

See [LICENSE.md](LICENSE.md) for full terms.

### Individual Skill Licenses

> ⚠️ **Important**: Each skill has its own license specified in the `license` metadata field within its `SKILL.md` file. These licenses may differ from the repository's MIT License and may include additional terms or restrictions. **Users are responsible for reviewing and adhering to the license terms of each individual skill they use.**

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=sacrrie/claude-geoai-skills&type=date&legend=top-left)](https://www.star-history.com/#sacrrie/claude-geoai-skills&type=date&legend=top-left)
