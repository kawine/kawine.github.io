# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based personal academic website hosted on GitHub Pages. The site serves as a portfolio for Kawin Ethayarajh, showcasing academic publications, CV, and professional information.

## Architecture

- **Static Site**: Pure HTML/CSS with Bootstrap 3.3.7 for responsive design
- **Jekyll Integration**: Uses Jekyll's `_config.yml` for GitHub Pages deployment
- **Asset Management**: Academic papers, slides, and media files stored in `/assets/` directory
- **Generated Content**: Jekyll builds the site into `/_site/` directory for deployment

## Key Files

- `index.html`: Main page containing complete site content (no separate templates)
- `_config.yml`: Jekyll configuration (minimal/empty)
- `/assets/`: Academic papers, CV, presentation slides, profile images
- `/_site/`: Jekyll build output (auto-generated, not for manual editing)

## Development Workflow

This is a Jekyll site that auto-deploys via GitHub Pages. No build commands or package managers are used.

### Making Changes
1. Edit `index.html` directly for content updates
2. Add new assets to `/assets/` directory
3. Commit changes - GitHub Pages will automatically rebuild and deploy

### Local Development
```bash
# If Jekyll is installed locally:
jekyll serve

# Or use GitHub Pages gem:
bundle exec jekyll serve
```

### Content Updates
- Profile information: Edit the sidebar section in `index.html:69-83`
- Main content: Edit the content section in `index.html:85-143`
- Publications: Update the nested lists in the "Notable work" section
- Styling: Modify the `<style>` block in `index.html:26-64`

## Important Notes

- Site uses Bootstrap 3.3.7 via CDN (older version for compatibility)
- Responsive design with desktop/mobile breakpoints at 992px
- Font Awesome 4.7.0 for icons
- No JavaScript dependencies beyond Bootstrap
- All content is in a single HTML file for simplicity