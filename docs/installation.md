# 🛠️ Installation Instructions for AFCCP

This guide walks you through setting up the **Air Force Cadet Career Problem (AFCCP)** repository using **PyCharm** and **Anaconda**. You’ll first clone the repository using terminal, then choose between two setup options:
- Using **Terminal**
- Using **PyCharm’s GUI**

---

## 📥 1. Install Required Tools

### ✅ Required:
- [Git](https://git-scm.com/downloads)
- [Anaconda](https://www.anaconda.com/products/distribution) (Python 3.8+)
- [PyCharm Community or Professional Edition](https://www.jetbrains.com/pycharm/download)

> ⚠️ We recommend the **Professional Edition** if you have access, but the **Community Edition** works fine too.

---

## 🌀 2. Clone the AFCCP Repository

Use your terminal (Mac/Linux: Terminal, Windows: Anaconda Prompt or Git Bash). First navigate to your desired
directory to clone this project. Personally, I have a folder called "Coding Projects" where I keep all of my python 
projects:

```bash
cd "/Users/griffenlaird/Coding Projects"
```

Once you're in the directory you want to clone afccp to, clone it!:

```bash
git clone https://github.com/dglaird/afccp.git
cd afccp
```

When you clone the repo, you should see some output that looks like this:

<pre>
```bash
$ git clone https://github.com/dglaird/afccp.git
Cloning into 'afccp'...
remote: Enumerating objects: 108, done.
remote: Counting objects: 100% (108/108), done.
remote: Compressing objects: 100% (82/82), done.
Receiving objects: 100% (108/108), 120.34 KiB | 2.35 MiB/s, done.
```
</pre>

---

## ⚙️ 3. Set Up the Python Environment

You have two options:

---

### 🧪 Option A: Using Terminal (Recommended for reproducibility)

#### 1. Create a new environment:

```bash
conda create -n afccp python=3.8 -y
```

#### 2. Activate the environment:

```bash
conda activate afccp
```

#### 3. Install all dependencies:

```bash
pip install -r requirements.txt
```

#### 4. Launch PyCharm, and **select this environment** when opening the project:
   File > Settings > Project: afccp > Python Interpreter > Add > Conda Environment > Existing environment

---

### 💻 Option B: Using PyCharm (GUI)

#### 1. Open **PyCharm** and select:
   - `Open Project` → navigate to and select the cloned `afccp/` directory.

#### 2. Go to:
   - `PyCharm > Preferences` (Mac) or `File > Settings` (Windows)

#### 3. Navigate to:
   - `Project: afccp > Python Interpreter`

#### 4. Click the ⚙️ gear icon → `Add Interpreter` → `Conda Environment`.

#### 5. Choose:
   - `New environment` (or Existing if you’ve already created one)
   - Python version: **3.8**

#### 6. Once created, click **OK**, and PyCharm will create and set up the interpreter.

#### 7. Now install the dependencies:
   - Open the built-in PyCharm Terminal
   - Run:

     ```bash
     pip install -r requirements.txt
     ```

---

## 🛠 4. Run an Example

Navigate to the main model script (adjust path if needed) (TODO: Need to modify this example)

```bash
python afccp/main.py
```

This should execute a full cadet-to-AFSC assignment using the default input data.

---

## 📁 5. Project Structure

```plaintext
afccp/
├── afccp/                # Core source code
├── docs/                 # MkDocs documentation
├── data/                 # Input instance data
├── requirements.txt      # Python dependencies
├── mkdocs.yml            # MkDocs config file
└── README.md             # Project overview
```

---

## 📚 6. View the Documentation Locally

If you want to view the full documentation site locally:

```bash
mkdocs serve
```

Then visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🚀 7. (Optional) Rebuild Reference Pages

To generate API reference pages:

```bash
python afccp/gen_ref_pages.py
```

> Be sure to run this from the **project root directory**, where `mkdocs.yml` is located.

---

## 🧪 8. Run Unit Tests (if available)

If the repo includes tests:

```bash
pytest tests/
```

---

## 🧠 More Info

- Full model explanation: [Model Overview](model/overview.md)
- API reference: [API Reference](api_reference.md)
- Author: [Griffen Laird](https://github.com/dglaird)

---

Happy modeling! ✈️