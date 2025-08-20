# 🔄 Migration Notice: Repository Restructuring for `rompy`

The `rompy` project is being restructured to improve clarity, modularity, and long-term maintainability.

We are splitting the monolithic repository into four smaller, purpose-driven repositories:

---

## 📦 New Repository Layout

| Repository | Purpose |
|-----------|---------|
| [`rompy`](https://github.com/rom-py/rompy) | 🧠 Core framework (remains at original location) |
| [`rompy-swan`](https://github.com/rom-py/rompy-swan) | 🌊 SWAN model components |
| [`rompy-schism`](https://github.com/rom-py/rompy-schism) | 🌊 SCHISM model components |
| [`rompy-notebooks`](https://github.com/rom-py/rompy-notebooks) | 📓 Examples, tutorials, workflows |

---

## 🗂️ Migration Plan

1. **Rename the current `rompy` repository**  
   - The existing monolithic repo will be renamed to `rompy-legacy`.
   - All code, history, issues, and PRs will remain in `rompy-legacy`.

2. **Create a new, slimmed-down `rompy` repo**  
   - A new repository named `rompy` will be created at the original URL.
   - It will contain only the clean, minimal core library.

3. **Spin out model integrations and notebooks**  
   - SWAN-related code → `rompy-swan`  
   - SCHISM-related code → `rompy-schism`  
   - Tutorials/examples → `rompy-notebooks`

4. **Cross-link all repos**  
   - Each repo will contain a README banner pointing to the others.

---

## 🔁 GitHub Redirect Behavior

GitHub automatically sets up redirects when a repository is renamed:

- **Existing clones of `rompy` will transparently point to `rompy-legacy`.**
- **Pulls and pushes will be redirected to `rompy-legacy` — safely.**
- **No one will accidentally overwrite the new `rompy` repo.**
- Issues, PRs, commit hashes, and tags remain valid under the new name.

> ✅ **You do *not* need to change your local git remotes.**  
> Your clone of `rompy` will now be pointing to `rompy-legacy` automatically.

> ⚠️ **Do not update your `origin` to the new `rompy` repo unless you explicitly want to work on the new core.**  
> Otherwise, you may push unrelated history to the new slimmed-down repo.

---

## 🧠 Why This Approach?

### ✅ Safe for existing users
- Users with existing checkouts keep working on the full-featured legacy repo.
- They don’t need to change anything.
- Pushes/pulls still work — but go to `rompy-legacy`.

### 🧭 Clear separation of concerns
- `rompy` → clean, minimal, core logic
- `rompy-swan` → SWAN model components
- `rompy-schism` → SCHISM model components
- `rompy-notebooks` → examples and workflows

### 🧬 Preserves history
- Full development history is retained in `rompy-legacy`.
- New repos can retain relevant history using `git filter-repo` or `git subtree`.

### 🛠 Easier maintenance
- Each repository can evolve independently.
- CI, releases, and issues are more focused.
- Contributors can work on what matters to them without navigating an oversized codebase.

---

## 🧑‍💻 For Contributors

If you want to:
- 🧾 **Continue working with the full legacy code** → keep using your current clone. It now points to `rompy-legacy`.
- 🧠 **Start working on the new slimmed-down core** → clone the new `rompy`:
  ```bash
  git clone https://github.com/rom-py/rompy.git

