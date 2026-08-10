# Publishing these wiki pages

GitHub wikis are a separate git repository from the code. These files are kept
here so they are versioned alongside the tracker, then pushed to the wiki repo.

## First time

Create the wiki by adding one page through the GitHub web UI (Wiki tab → Create
the first page → Save). GitHub only creates the wiki repository once a page
exists; cloning before that fails.

Then, from the tracker repo root:

```bash
git clone https://github.com/StefanMPopp/tracking.wiki.git /tmp/tracking-wiki
cp wiki/*.md /tmp/tracking-wiki/
cd /tmp/tracking-wiki
git add -A
git commit -m "Add documentation pages"
git push
```

## Updating

After editing anything in `wiki/`:

```bash
cd /tmp/tracking-wiki && git pull
cp ~/tracking/wiki/*.md .
git add -A && git commit -m "Update docs" && git push
```

## Notes

- `_Sidebar.md` renders as the navigation panel on every page.
- Page filenames become URLs: `Starting-a-new-project.md` → `.../wiki/Starting-a-new-project`.
  Renaming a file breaks inbound links.
- Links between pages use the bare page name: `[Installation](Installation)`.
- The main `README.md` links to the wiki with relative `../../wiki/Page` paths,
  which work from any fork without hardcoding the repo owner.
