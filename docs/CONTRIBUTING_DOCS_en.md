# Documentation Contribution Guide

We welcome your contributions to the project documentation. High-quality documentation is crucial for project success. This guide will help you efficiently submit documentation that meets the standards.

## Contribution Scope

We welcome any contributions that can improve documentation quality, including but not limited to:

- Correction and Improvement: Fix typos, grammar errors, incorrect code examples, outdated information, or broken links.

- Clarification and Optimization: Make descriptions clearer and easier to understand, optimize sentence structure, and supplement background knowledge.

- Content Supplement: Add usage examples, API documentation, frequently asked questions (FAQ), best practices, or warning descriptions for existing features.

- New Content Creation: Write new chapters or tutorials for newly added features, such as operator README, API introduction documents, and so on. If you have questions, we recommend creating an Issue for discussion first.

- Localization Translation: Help us translate or proofread documents in other languages.

- Style and Navigation: Improve the layout, readability, and navigation structure of the documentation website.

## Contribution Process

1. **Preparation Work**

    - Determine the Task: If there are documentation issues, you can create new Issues. We recommend using the label category `[Documentation|文档反馈]` and providing a detailed description. Based on the existing Issues list, determine the documentation issues to be resolved.
    - Claim the Task: Comment `/assign @yourself` under the corresponding Issue to indicate that you will handle it and avoid duplicate work.

2. **Document Modification**

    - Select Branch: Please download the source code from the master or other Tag branches to the local machine.
    - Follow Format:
      - This project recommends using **Markdown format**.
      - Follow the existing writing style of the project.
      - Put static resources such as images in the corresponding directory. For example, images are generally in the `figures` folder under the docs directory. You can adjust them yourself in special cases.
    - Careful Addition and Deletion: When modifying content, please try to maintain the original line width and line break conventions.

3. **Submit Changes**

    - Atomic Commit: Each commit should focus on an independent modification. For example, "Fix spelling errors in xx guide" and "Update example code in API reference" should be submitted separately.

    - Write Clear Commit Messages:

      ```text
      Brief description (no more than 50 characters)

      If necessary, provide a more detailed description here. Explain the reason and content of the modification, rather than what specifically was changed (the code itself will show).
      Associated Issue: #123
      ```

4. **Initiate Pull Request**

    - Target Branch: Please merge the PR into the target branch of the project.
    - Title and Description:
      - PR Title: Should clearly summarize the modification, for example: `[Docs] Fix configuration example in quick start`.
      - PR Description: Detailed explanation of your changes, motivation, and associated Issues (use Closes #123 or Fixes #456).
    - Preview Check: Please check the document effect in local or online browsing in advance to ensure that the rendering meets expectations.
    - Wait for Review: Maintainers will review and may propose modification suggestions. Please follow up on the discussion in a timely manner.

## Writing Standards

Before developers write project documentation, please be sure to read the following standards first. If you have questions, you are welcome to make suggestions at any time!

- Prerequisites: Please first learn the unified writing standards provided by the CANN organization. For details, see [CANN Document Writing Standards](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md).

  - Document Content Requirements: Introduce the required and optional document deliverables in the project.
  - Directory Structure Standards: Introduce the principles of directory division, such as Chinese and English management.
  - Content Element Standards: Introduce rules for different writing elements, such as file naming, titles, fonts, images, code blocks, links, and so on.

- Precautions:

  In addition to the above writing rules, you also need to pay attention to the following:

  - Tone: Use a friendly, professional, and neutral tone. For beginners, avoid unnecessary jargon.
  - Terminology: Maintain terminology consistency (such as uniformly using "click" instead of "single click"). Please refer to the project terminology table (if available).
  - Code Examples:
    - Ensure that all code examples are runnable and tested.
    - Provide sufficient context and explanation.
    - Indicate the environment or prerequisites required for code running.
  - Punctuation and Format:
    - When mixing Chinese and English, use full-width punctuation. Punctuation marks must conform to the Chinese/English context.
    - Use appropriate hierarchy for titles (#, ##, ###).
    - Use lists and tables to organize complex information.
  - Links: Use descriptive link text, avoid "click here", and ensure that link resources are authentic and reliable.
  - Images:
    - Common Formats: We recommend the png format. Try to keep the style consistent with existing images.
    - Resolution and Clarity: Must be clear and of moderate size. Avoid blurring or excessive compression.
    - File Size: We do not recommend that a single image exceeds 10M.
  - Copyright: For all quoted images, literature, and other resources, please ensure compliance.

## Get Help

If you have any questions during the contribution process:

1. Check Existing Documentation: If there are problems with templates or standards, please first check the existing guides, API documentation, or README of the project.
2. Initiate Discussion: You can create a new Issue or leave a message directly in the relevant Issue or PR.

## Document Templates

The key documents involved in operator deliverables mainly include the following. For specific writing formats and content requirements, please refer to the templates.

- [Operator README Document Template](https://gitcode.com/cann/ops-ras/wiki/%E7%AE%97%E5%AD%90README%E6%96%87%E6%A1%A3%E6%A8%A1%E6%9D%BF)
- [aclnn API Document Template](https://gitcode.com/cann/ops-ras/wiki/aclnn%20API%E6%96%87%E6%A1%A3%E6%A8%A1%E6%9D%BF)
