# 🧠 PyCharm Setup Guide for AFCCP

This guide walks you through setting up **PyCharm** for use with the **Air Force Cadet Career Problem (AFCCP)** project. It covers:
- Opening the project
- Setting up the Python interpreter
- Creating a run configuration
- Using the built-in terminal

---

## 🧰 1. Install PyCharm

If you haven't already:

- Download from [https://www.jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download)
- Install either the **Community** (free) or **Professional** edition

---

## 📂 2. Open the Project Folder

1. Open PyCharm.
2. Click **"Open"** on the welcome screen.
3. Navigate to the directory where you cloned the `afccp` repository (e.g., `~/Documents/Projects/afccp`) and click **Open**.

> 📝 If you haven’t cloned the repository yet, do this from terminal first:
> ```bash
> git clone https://github.com/dglaird/afccp.git
> cd afccp
> ```

---

## 🐍 3. Configure the Python Interpreter (Conda or VirtualEnv)

1. Go to **File > Settings > Project: afccp > Python Interpreter**
2. Click the ⚙️ icon (top-right) and choose **Add...**
3. In the dialog:
   - Select **Conda Environment**
   - Choose:
     - **Existing environment** if you've already created one (e.g., via terminal):
       ```bash
       conda create -n afccp python=3.8 -y
       conda activate afccp
       ```
     - **New environment** if you want PyCharm to create it
   - Set Python version to **3.8**
4. Click **OK** to save and apply the interpreter

> ✅ Once set, PyCharm will index your environment and show packages in the interpreter panel.

---

## 📦 4. Install Project Dependencies

Once your interpreter is set:

1. Open the **Terminal** tab at the bottom of PyCharm
2. Run:

```bash
pip install -r requirements.txt
```

This installs all packages required for AFCCP, including `pyomo`, `pandas`, `mkdocs`, and more.

---

## ▶️ 5. Create a Run Configuration

To quickly run the model script from inside PyCharm:

1. Go to **Run > Edit Configurations...**
2. Click the ➕ to add a new configuration
3. Choose **Python**
4. Set:
   - **Name**: `Run AFCCP`
   - **Script path**: `afccp/example.py` (use the file chooser)
   - **Python Interpreter**: select the environment you just configured
5. Click **Apply** and **OK**

Now you can run the model with the ▶️ button in the top-right corner.

---

## 🧪 6. (Optional) Add Test Configurations

If you have test files in `tests/`, create another Run Configuration:

1. Add a new **pytest** configuration
2. Point it to the `tests/` directory
3. Give it a name like `Run Tests`

---

## 📁 7. Recommended PyCharm Settings

### Editor Settings
- Enable **soft wrap** for Markdown and `.py` files:  
  `Preferences > Editor > General > Soft Wraps`

### Markdown Preview
- `Preferences > Languages & Frameworks > Markdown > Preview`  
  Choose **"Preview with HTML and CSS"**

### Auto-save on Run
- `Preferences > Appearance & Behavior > System Settings > Save files on frame deactivation`

---

## 🧠 You're Ready to Develop!

You now have a fully working setup in PyCharm:
- Code highlighting and linting
- Conda environment
- One-click model execution
- Markdown editing with preview
- Git integration

Happy modeling!