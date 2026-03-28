---
name: ascend-doc-auto-updater
description: Detect and fetch the latest “Ascend C算子开发” PDF from the Huawei Ascend CANN Community Edition documentation, download page for a user-specified version, then convert the PDF into chapter-level markdown files and save them under ascend_dev_guide_sections/.
metadata: Auto-download AscendC operator dev PDF and export TOC-split Markdown.

---

# Ascend Doc Auto Updater Skill

## Purpose

When the user requests an update for a specific **CANN Community Edition** version (e.g. `850`, `900beta1`, `83RC1`), this skill will:

1. Build the CANN documentation **download page** URL for that version
2. Run `run.py` locally with that URL
3. Let `run.py` automatically:

   * Locate the PDF URL for **“Ascend C算子开发”**
   * Download the PDF
   * Split content by TOC into section Markdown files
   * Write all Markdown files to `ascend_dev_guide_sections/`

## Inputs (User Triggers)

The user may trigger this skill with phrases like:

* `更新到 850`
* `更新版本：900beta1`
* `拉取 83RC1 的 Ascend C算子开发文档`
* `更新 CANN 社区版 850`
* Or provide the full download page URL directly

## Output Folder

* `ascend_dev_guide_sections/`
  Each chapter/section is exported as one `.md` file.

## Version Parsing Rules

1. If the user provides a full URL that matches:

   * `.../CANNCommunityEdition/<version>/index/download`
     then **use it directly**.

2. Otherwise extract `<version>` from the user message:

   * Prefer the token that looks like a CANN CE version identifier, such as:

     * `850`, `860`, `900beta1`, `83RC1`
   * If multiple different version tokens exist, choose the one that is:

     * Closest to keywords like `CANN`, `CommunityEdition`, `社区版`, `版本`, `更新`
     * Otherwise choose the first best match

## Download Page Construction

If a version token is extracted, construct:

`download_page = https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/{version}/index/download`

## Execution Contract (Required Command)

When responding, the skill MUST provide:

1. The resolved `version`
2. The resolved `download_page`
3. A single shell command to run the update (include `--force`):

```bash
python run.py --download-page "<download_page>" --target-title "Ascend C算子开发" --force
```

## Notes / Assumptions

* `run.py` must support CLI args:

  * `--download-page` (required)
  * `--target-title` (default `"Ascend C算子开发"`)
  * `--force` (recommended to re-export even if the PDF URL/content is unchanged)
* The script is responsible for discovering the actual PDF URL from the download page and performing conversion.
* If the user input is ambiguous (e.g., multiple versions mentioned), prefer the version closest to “更新/版本/CANN/社区版” context; otherwise pick the first best match and proceed.

## Examples

### Example 1

User: `更新到 850`
Skill output:

* version: `850`
* download_page: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/index/download`
* command:

```bash
python run.py --download-page "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/index/download" --target-title "Ascend C算子开发" --force
```

### Example 2

User: `用这个链接更新：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/index/download`
Skill output:

* version: `900beta1`
* download_page: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/index/download`
* command:

```bash
python run.py --download-page "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/index/download" --target-title "Ascend C算子开发" --force
```
