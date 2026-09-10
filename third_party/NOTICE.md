# Upstream Attribution and Publication Review

This file records provenance concerns. It does not assign a license to the author's entire SPARK project or certify redistribution compliance.

| Upstream | Relevance | Reviewed source |
| --- | --- | --- |
| OpenAI CLIP | Frozen text encoder, text representations | [MIT license](https://github.com/openai/CLIP/blob/main/LICENSE); local copy: `CLIP_LICENSE.txt` |
| Fed-WSVAD | Existing local implementation and Event-based partition provenance | [Official repository](https://github.com/wbfwonderful/Fed-WSVAD); no repository-wide license file was shown in the reviewed root listing |
| FedTPG | `PreNorm`, `GEGLU`, `FeedForward`, `CrossAttention`, `SelfAttention`, and related prompt structures resemble the existing local prompt implementation | [Upstream prompt source](https://github.com/boschresearch/FedTPG/blob/main/model/prompt_net.py), copyright (c) 2024 Robert Bosch GmbH; [AGPL-3.0 license](https://github.com/boschresearch/FedTPG/blob/main/LICENSE) |
| VadCLIP | Feature-download provenance and upstream VAD implementation reference | [Official repository](https://github.com/nwpu-zxr/VadCLIP); [Apache-2.0 license](https://github.com/nwpu-zxr/VadCLIP/blob/main/LICENSE) |

## Review Before Distribution

FedTPG's prompt source explicitly carries an AGPL notice. Sections 5 and 6 of that license address modified works and non-source distribution, including Corresponding Source obligations. Whether and how these terms apply to this release requires checking the actual reused material, provenance, and any separately obtained permissions. Merely tracing/freezing a module or removing debug metadata is not an exemption.

Do not interpret the omission of training/export sources, the retained CLIP MIT notice, or this checklist as authorization for a closed-source distribution. Confirm the upstream permissions and the relevant source/notice obligations before making this candidate public. No legal conclusion about the entire SPARK project has been made here.

## 中文提示

现有项目引用了 Fed-WSVAD、VadCLIP 和 FedTPG。本地提示模块中的多个类与 FedTPG 的对应实现高度相似，而 FedTPG 的该文件明确标注 AGPL。需要先核查实际沿用范围及是否已有单独授权，不能把“只发二进制模型”当成可以忽略许可证的依据。

另外，当前检查的 Fed-WSVAD 根目录没有显示统一许可证文件；请确认所沿用代码、划分文件和标注的发布权限。本文只是发布前核查提示，不是对整个项目的法律认定，也没有替作者选择许可证。
