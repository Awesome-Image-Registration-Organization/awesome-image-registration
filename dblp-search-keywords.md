# DBLP 图像配准/匹配/点云配准检索关键词建议

> 本文档用于补充现有 `registra` 检索词未能覆盖的相关论文。
> 核心原则：**新增检索词与 `registra` 尽量互斥**，即尽量抓取标题不含 `registration` / `registering` / `registrations` 等词根、但实质属于配准/匹配领域的论文。

---

## DBLP 搜索语法速查

| 功能 | 语法 | 示例 | 说明 |
|------|------|------|------|
| 前缀搜索（默认） | 直接输入 | `registra` 匹配 `registration`, `registering` | **注意**：前缀不匹配更短的词根，如 `registra` **不匹配** `register` |
| 精确单词 | 末尾加 `$` | `graph$` 只匹配 `graph`，不匹配 `graphics` | 用于消除歧义 |
| 布尔 AND | 空格分隔 | `codd model` 表示同时包含两者 | 默认连接符 |
| 布尔 OR | `\|` 分隔 | `graph\|network` 表示包含任一 | 用于扩展同义词 |

> ⚠️ 短语搜索运算符 (`.`) 已禁用；布尔 NOT 运算符 (`-`) 已禁用。

### 语法陷阱：AND / OR 优先级

DBLP 中 **`|` 的优先级高于空格（AND）**，且**不支持括号**改变优先级。

- ❌ 错误写法：`image|feature match`  
  实际解析为：`image* OR (feature* AND match*)`  
  会返回所有含 "image" 的论文（无论是否含 match），噪声巨大。

- ✅ 正确写法：`image match|feature match`  
  实际解析为：`(image* AND match*) OR (feature* AND match*)`  
  这才是我们想要的语义。

**结论**：当多个 AND 组合需要用 `|` 连接时，必须把公共部分重复写出，不能省略。

---

## 与 `registra` 互斥的检索词

### 一、高互斥度（强烈推荐）

这些方向的论文标题**几乎不用** `registra` 词根，但与图像配准/点云配准属于同一技术范畴或强相关领域。

#### 1. 图像匹配（Image Matching）

`registra` 最大的盲区。匹配（Matching）与配准（Registration）是两个平行发展的社区，**标题用词高度分化**。

| 检索词 | DBLP 语法 | 覆盖内容 |
|--------|-----------|---------|
| 图像匹配 | `image match` | image matching / image match |
| 特征匹配 | `feature match` | feature matching / feature matcher |
| 关键点匹配 | `keypoint match` | keypoint matching |
| 图匹配 | `graph match` | graph matching |
| 立体匹配 | `stereo match` | stereo matching |

**组合检索式：**
```text
image match|feature match|keypoint match|graph match|stereo match
```

> 覆盖示例：*ASpanFormer: Detector-Free Image Matching*, *3DG-STFM: 3D Geometric Guided Student-Teacher Feature Matching*, *Semi-Supervised Keypoint Detector and Descriptor for Retinal Image Matching*

#### 2. 光流 / 场景流（Optical Flow / Scene Flow）

光流估计本质上就是**稠密可变形图像配准**，但两个社区用词完全不同。

| 检索词 | DBLP 语法 |
|--------|-----------|
| 光流 | `optical flow` |
| 场景流 | `scene flow` |

**组合检索式：**
```text
optical flow|scene flow
```

> 与 `registra` 交集极低，技术内涵直接对应 deformable registration。

#### 3. 图像拼接 / 镶嵌（Image Stitching / Mosaicking）

图像拼接的核心前提就是多图配准，但该领域论文标题只用 `stitching/mosaic`，**从不用** `registration`。

| 检索词 | DBLP 语法 |
|--------|-----------|
| 拼接 | `stitch` |
| 全景图 | `panorama` |
| 镶嵌 | `mosaic` |

**组合检索式：**
```text
stitch|panorama|mosaic
```

#### 4. 点云 ICP（Iterative Closest Point）

ICP 是点云配准最经典算法，大量 ICP 变体论文标题**只提 ICP** 而不写 `registration`。

| 检索词 | DBLP 语法 |
|--------|-----------|
| ICP | `ICP` |

**检索式（建议拆分为两次独立查询）：**
```text
ICP
iterative closest
```

> 覆盖示例：*Generalized-ICP*, *Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration*
>
> ⚠️ **注意**：`ICP|iterative closest` 在部分 venue 下会返回 0 结果（如 `ICP|iterative closest venue:CVPR:` 返回 0，但 `ICP venue:CVPR:` 单独有 7 篇）。建议拆分为两个独立查询分别调用。

---

### 二、中等互斥度（推荐作为补充）

这些检索词**大部分结果与 `registra` 互斥**，但存在少量交集，适合分主题补充检索。

#### 5. 对应关系 / 对齐（Correspondence / Alignment）

| 检索词 | DBLP 语法 | 备注 |
|--------|-----------|------|
| 对应关系 | `correspond` | 纯对应关系论文通常不含 `registra`；少数论文同时出现 |
| 对齐 | `align` | 部分论文用 alignment 替代 registration，有少量重叠 |

**组合检索式：**
```text
correspond|align
```

#### 6. 医学图像相关（Medical Image）

图谱构建、运动校正等任务常隐含配准，但标题习惯不同。

| 检索词 | DBLP 语法 |
|--------|-----------|
| 图谱构建 | `atlas construction` |
| 运动校正 | `motion correction` |
| 变形场 | `deformation field` |

**检索式（建议拆分为三次独立查询）：**
```text
atlas construction
motion correction
deformation field
```

> ⚠️ **注意**：`atlas construction|motion correction` 全局理论并集应为 800+ 篇，但 DBLP 实际仅返回 2 篇；加 venue 过滤后甚至返回 0。低频双字词组的 `|` 组合存在严重结果丢失，建议完全拆分为独立查询。

---

## ⚠️ 为什么不能简写？一个具体例子

以图像匹配检索式为例：

| 写法 | DBLP 实际解析 | 语义 | 是否可用 |
|------|--------------|------|---------|
| `image match\|feature match\|keypoint match` | `image* match* \| feature* match* \| keypoint* match*` | 含 (image+match) 或 (feature+match) 或 (keypoint+match) | ✅ |
| `image\|feature\|keypoint match` | `image* \| feature* \| keypoint* match*` | 含 image **或** feature **或** (keypoint+match) | ❌ |

第二种写法会引入大量只含 "image" 或 "feature" 但不含 "match" 的无关论文，因此**不可省略重复词**。

> 这一规则适用于本文档中所有包含 `\|` 的检索式，如 `optical flow\|scene flow` 等。对于低频双字词组（如 `atlas construction\|motion correction`），即使只有两个分支，也可能严重丢失结果，建议拆分为独立查询。

---

## 可直接使用的 DBLP 查询字符串

### 分主题检索（推荐，低噪声）

复制以下表达式直接粘贴到 DBLP 搜索框即可：

| 主题 | DBLP 查询字符串 |
|------|----------------|
| 🎯 图像匹配 | `image match\|feature match\|keypoint match\|graph match\|stereo match` |
| 🌊 光流/场景流 | `optical flow\|scene flow` |
| 🧩 图像拼接 | `stitch\|panorama\|mosaic` |
| ☁️ 点云 ICP | `ICP` / `iterative closest`（两次独立查询） |
| 🔗 对应关系/对齐 | `correspond\|align` |
| 🏥 医学图像 | `atlas construction` / `motion correction` / `deformation field`（三次独立查询） |

---

## 多组互补检索式（API 调用推荐）

> 由于 DBLP API 对单个查询的长度和复杂度存在限制（URL 长度、解析深度等），**不建议将所有检索词堆砌为单个查询**。建议按以下多组分次调用 API，最后合并去重。

### 分组原则

- **组内同质**：每组聚焦一个技术方向，内部用 `|` 扩展同义词。
- **组间互补**：各组之间无重叠覆盖，合起来覆盖全部互斥方向。
- **长度可控**：每组长度适中，避免 API 拒绝或超时。

### 推荐分组（共 8 个方向，10 次查询）

| 组号 | 技术方向 | DBLP 查询字符串 | 说明 |
|------|---------|----------------|------|
| **G1** | 图像匹配 | `image match\|feature match\|keypoint match\|graph match\|stereo match` | 覆盖图像/特征/关键点/图/立体匹配 |
| **G2** | 光流与场景流 | `optical flow\|scene flow` | 稠密可变形配准的等价社区 |
| **G3** | 图像拼接与镶嵌 | `stitch\|panorama\|mosaic` | 多图配准相关 |
| **G4a** | 点云 ICP | `ICP` | 点云配准经典算法（ICP 缩写） |
| **G4b** | 点云 ICP（扩展） | `iterative closest` | 点云配准经典算法（全称） |
| **G5** | 对应关系与对齐 | `correspond\|align` | 对应估计与对齐 |
| **G6a** | 医学图像（图谱） | `atlas construction` | 图谱构建 |
| **G6b** | 医学图像（校正） | `motion correction` | 运动校正 |
| **G6c** | 医学图像（变形场） | `deformation field` | 变形场估计 |

> **为什么 G6 拆分为三组？** 实测发现 `atlas construction\|motion correction\|deformation field`（三个双字词组用 `\|` 连接）在 DBLP 中返回 **0 结果**；进一步测试发现即使只有两个低频双字词组（`atlas construction\|motion correction`），全局理论并集 800+ 篇也被压缩为仅 2 篇。因此三个词全部拆分为独立查询。详见下文"DBLP 已知限制"。

### 与 venue 过滤结合的分组示例

如果只需要特定会议的论文，可在每组后追加 venue 过滤器：

| 目标 | G1 (图像匹配) + CVPR | G4 (点云 ICP) + CVPR |
|------|----------------------|----------------------|
| 查询 | `image match\|feature match\|keypoint match venue:CVPR:` | `ICP venue:CVPR:` / `iterative closest venue:CVPR:` |

### 综合检索式（仅作参考，不推荐用于 API）

如果你坚持单次查询，可使用以下完整表达式（长度较长，可能受 API 限制）：

```text
image match|feature match|keypoint match|graph match|stereo match|optical flow|scene flow|stitch|panorama|mosaic|ICP|iterative closest|correspond|align|atlas construction|motion correction|deformation field
```

> ⚠️ 综合式噪声较高且可能触发长度限制，**强烈建议使用上方 G1-G6 分组方案**。特别注意：`ICP|iterative closest` 和 `atlas construction|motion correction` 等低频组合在综合式中问题更严重。

---

## DBLP 已知限制（实测发现）

### 限制 1：低频双字词组的 `\|` 组合严重丢失结果

**现象**：当使用 `\|` 连接多个分支，且分支为**低频双字词组**（每个分支内部包含空格 AND）时，DBLP 会**严重丢失结果**，甚至返回 **0 结果**。

**实测验证**：

| 查询 | 解析结果 | 返回结果数 | 状态 |
|------|---------|-----------|------|
| `atlas construction\|motion correction\|deformation field` | `atlas* construction* \| motion* correction* \| deformation* field*` | **0** | ❌ |
| `atlas construction\|motion correction` | `atlas* construction* \| motion* correction*` | **2**（理论并集 800+） | ❌ |
| `atlas construction` | `atlas* construction*` | 159 | ✅ |
| `motion correction` | `motion* correction*` | 714 | ✅ |
| `deformation field` | `deformation* field*` | 329 | ✅ |
| `ICP\|iterative closest venue:CVPR:` | — | **0** | ❌ |
| `ICP venue:CVPR:` | — | 7 | ✅ |
| `iterative closest venue:CVPR:` | — | 0 | ✅（确实无结果） |
| `atlas construction\|motion correction\|deformation` | `atlas* construction* \| motion* correction* \| deformation*` | 5 | ✅ |

**规律**：
- ❌ `A B\|C D\|E F`（三个双字词组）→ 可能返回 0
- ❌ `A B\|C D`（**两个低频双字词组**）→ **严重丢失结果**（如 800+ → 2）
- ❌ `A B\|C D`（两个双字词组 + venue 过滤）→ 可能返回 0
- ✅ `A B\|C D`（两个**高频**双字词组，如 `image match\|feature match`）→ 正常
- ✅ `A B\|C D\|E`（两个双字 + 一个单字）→ 正常
- ✅ `A\|B\|C`（三个单字词）→ 正常
- ✅ `A B\|C D\|E F\|G`（含单字词穿插）→ 正常

**结论**：
- 若多个 `(word1 word2)` 形式的分支需要 OR 连接，**建议最多两个双字词组连用**，且确保它们是**高频词组**。
- 对于**低频词组**（如 `atlas construction`、`motion correction`、`iterative closest`），**无论几个分支，均建议拆分为多次独立 API 调用**，避免结果丢失。

---

## 与 venue 过滤结合使用

你可以将上述检索词与 DBLP 的 venue/type 过滤器结合，精确锁定目标会议/期刊。

| 目标 | DBLP 查询字符串示例 |
|------|---------------------|
| CVPR 图像匹配 | `image match|feature match venue:CVPR:` |
| ICCV 光流 | `optical flow venue:ICCV:` |
| MICCAI 医学图像（图谱） | `atlas construction venue:MICCAI:` |
| MICCAI 医学图像（校正） | `motion correction venue:MICCAI:` |
| TPAMI 期刊论文 | `ICP type:Journal_Articles: venue:IEEE_Trans._Pattern_Anal._Mach._Intell.:` |

> venue 和 type 过滤器的写法可参考 README 中已有的 DBLP 链接格式。

---

## 互斥性验证方法

你可以快速验证新增检索词与 `registra` 的互斥程度：

1. 搜索 `registra`，记录结果数 **N1**
2. 搜索新增词（如 `image match|feature match`），记录结果数 **N2**
3. 搜索交集 `registra image match|registra feature match`，记录结果数 **N3**
4. **新增独有率 ≈ (N2 - N3) / N2**，该值越高说明互斥性越强

---

## 附：关于 `registra` 前缀覆盖范围的补充说明

DBLP 的默认搜索是**前缀匹配**，这意味着：

- `registra` 匹配：`registration`, `registering`, `registrations`, `registrable` 等
- `registra` **不匹配**：`register`, `registered`, `registers`

如果你的目标是**最大化覆盖**所有配准相关论文，建议在现有 `registra` 的基础上，额外补充 `register`：

```text
registra|register
```

> 但本文档的核心目标是**与 `registra` 互斥的新增检索词**，因此上述补充不在主要推荐之列，仅作备注。
