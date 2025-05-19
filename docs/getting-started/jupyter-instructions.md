# 🧪 Jupyter Notebook Setup for AFCCP

This tutorial walks you through setting up the **afccp environment** to work with **Jupyter Notebooks**, 
using Anaconda or Miniconda to manage dependencies.

---

## 📦 1. Install Anaconda (or Miniconda)

Before anything else, install a Python environment manager:

- **Anaconda (recommended):**  
  https://www.anaconda.com/products/distribution  
  *(Includes Python, Jupyter, and many scientific packages out-of-the-box)*

- **Miniconda (lightweight version):**  
  https://docs.conda.io/en/latest/miniconda.html

After installation, you should be able to run `conda` from your terminal or Anaconda Prompt.

---

## 🧪 2. Install Jupyter and Configure the Kernel

If you've followed the main [Installation Guide](../getting-started/installation.md), you should already have a 
working `afccp` virtual environment which will have the `notebook` and `ipykernel` packages. But, you can
always directly install them here. Regardless, you'll need to make the `afccp` environment available as a kernel:

```bash
pip install notebook ipykernel
python -m ipykernel install --user --name afccp --display-name "Python (afccp)"
```

This will let you choose **Python (afccp)** as a kernel option inside Jupyter.

---

## 🚀 3. Launch Jupyter Notebook

With the environment still activated, launch Jupyter:

```bash
jupyter notebook
```

This will open a browser window where you can create and run notebooks. Select **"Python (afccp)"** as your kernel.

---

## ✅ 4. Verify afccp Imports

Create a new notebook and run this code to test your setup:

```python
from afccp import CadetCareerProblem
print("AFCCP module loaded successfully!")
```

---

## 💡 Bonus: Recommended Notebook Extensions (Optional)

If you'd like a better notebook experience, install:

```bash
pip install jupyterlab jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```

These add features like collapsible cells, codefolding, and table of contents sidebars.

---

## 🧠 You're All Set!

You now have a fully configured AFCCP development environment running inside Jupyter. Use it to:
- Explore AFCCP models interactively
- Visualize value functions and assignments
- Run analyses and share notebooks with others

You may now return to [Tutorial 1](../user-guide/tutorial_1.md) to resume the introduction.