# -*- coding: utf-8 -*-
"""Generate plain ppt from 20260702.md (white bg, black text)."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

CN_FONT = "Microsoft YaHei"
BLACK = RGBColor(0x00, 0x00, 0x00)
GREY = RGBColor(0x80, 0x80, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
blank = prs.slide_layouts[6]


def set_font(run, size=18, bold=False, color=BLACK, font=CN_FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("eastAsia", "cs"):
        el = rPr.find(qn(f"a:{tag}"))
        if el is None:
            el = etree.SubElement(rPr, qn(f"a:{tag}"))
        el.set("typeface", font)


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=BLACK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    set_font(r, size=size, bold=bold, color=color)
    return tb


def add_title(slide, title):
    add_text(slide, Inches(0.6), Inches(0.4), Inches(12.1), Inches(0.7),
             title, size=28, bold=True)
    # thin underline
    line = slide.shapes.add_connector(1, Inches(0.6), Inches(1.15),
                                      Inches(12.7), Inches(1.15))
    line.line.color.rgb = BLACK
    line.line.width = Pt(1)


def add_paragraphs(slide, x, y, w, h, items, size=18, gap=8, indent_marker=True):
    """items: list of str or (head, tail). head bold, tail normal."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(gap)
        if indent_marker:
            r0 = p.add_run()
            r0.text = "• "
            set_font(r0, size=size, bold=False)
        if isinstance(it, tuple):
            head, tail = it
            r1 = p.add_run(); r1.text = head
            set_font(r1, size=size, bold=True)
            r2 = p.add_run(); r2.text = tail
            set_font(r2, size=size)
        else:
            r1 = p.add_run(); r1.text = it
            set_font(r1, size=size)


def add_numbered(slide, x, y, w, h, items, size=18, gap=8):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(gap)
        r0 = p.add_run(); r0.text = f"{i+1}. "
        set_font(r0, size=size, bold=True)
        if isinstance(it, tuple):
            head, tail = it
            r1 = p.add_run(); r1.text = head
            set_font(r1, size=size, bold=True)
            r2 = p.add_run(); r2.text = tail
            set_font(r2, size=size)
        else:
            r1 = p.add_run(); r1.text = it
            set_font(r1, size=size)


# =================== SLIDE 1: Title ===================
s = prs.slides.add_slide(blank)
add_text(s, 0, Inches(2.6), SW, Inches(1.0),
         "FB Pipeline", size=48, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_text(s, 0, Inches(3.6), SW, Inches(0.7),
         "从自然语言到可执行生产计划", size=28,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_text(s, 0, Inches(4.4), SW, Inches(0.5),
         "LLM 驱动的结构化提取 + 求解器排期", size=18, color=GREY,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_text(s, 0, Inches(6.6), SW, Inches(0.4),
         "2026 / 07 / 02", size=14, color=GREY,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# =================== SLIDE 2: 研究问题 ===================
s = prs.slides.add_slide(blank)
add_title(s, "1. 研究问题")

add_text(s, Inches(0.6), Inches(1.5), Inches(12.1), Inches(0.5),
         "前置工作（论文 2510.02679）提出的问题：", size=18, bold=True)
add_text(s, Inches(1.0), Inches(2.15), Inches(11.7), Inches(1.5),
         "如何从自然语言（NL）工艺描述中提取出结构化的工艺流程，\n"
         "以约束的形式喂给求解器，自动完成工厂排期并生成可执行的生产计划。",
         size=18)

add_text(s, Inches(0.6), Inches(4.3), Inches(12.1), Inches(0.5),
         "FB Pipeline 的立足点：", size=18, bold=True)
add_text(s, Inches(1.0), Inches(4.95), Inches(11.7), Inches(1.8),
         "在前置工作基础上，重新思考\n"
         "「NL → 结构化 → 约束 → 排期 → 生产计划」这条信息流上，\n"
         "LLM 与代码各自应当负责的部分。",
         size=18)


# =================== SLIDE 3: 信息流 & 对比 ===================
s = prs.slides.add_slide(blank)
add_title(s, "2. 整体信息流 & 与传统 / DSL 方法的对比")

# text-based flow
add_text(s, Inches(0.6), Inches(1.4), Inches(12.1), Inches(0.5),
         "NL 文本  →  结构化 JSON  →  约束矩阵 (or_matrix)  →  裸排期  →  可执行生产计划",
         size=16, bold=True, align=PP_ALIGN.CENTER)

# table
rows = [
    ["环节", "传统做法", "DSL 论文做法", "FB Pipeline（本工作）"],
    ["NL → 结构化", "人工录入 ERP / APS", "DSL 提取 + 翻译",
     "纯 LLM 提取 + 多步校验 + 归一化\n固定 schema + 额外字段"],
    ["结构化 → 约束", "人工 / 顾问配置规则", "PDA", "linear dependency"],
    ["约束 → 排期", "求解器", "求解器（OR-Tools）", "求解器（OR-Tools，相同）"],
    ["排期 → 可执行计划", "人工对照工艺单填工单",
     "代码自动回填，从 DSL 程序取信息",
     "代码自动回填，从固定 schema 取前后产物 / 参数"],
]
tx, ty = Inches(0.5), Inches(2.15)
tw, th = Inches(12.3), Inches(3.8)
rows_n, cols_n = len(rows), len(rows[0])
tbl = s.shapes.add_table(rows_n, cols_n, tx, ty, tw, th).table
col_widths = [Inches(2.0), Inches(2.7), Inches(3.0), Inches(4.6)]
for i, cw in enumerate(col_widths):
    tbl.columns[i].width = cw
tbl.rows[0].height = Inches(0.55)
for r in range(1, rows_n):
    tbl.rows[r].height = Inches(0.82)

for r in range(rows_n):
    for c in range(cols_n):
        cell = tbl.cell(r, c)
        cell.margin_left = Inches(0.12)
        cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.06)
        cell.margin_bottom = Inches(0.06)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.fill.solid(); cell.fill.fore_color.rgb = WHITE
        tf = cell.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].text = ""
        lines = rows[r][c].split("\n")
        for li, ln in enumerate(lines):
            p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.LEFT
            run = p.add_run()
            run.text = ln
            if r == 0:
                set_font(run, size=13, bold=True, color=BLACK)
            elif c == 0:
                set_font(run, size=12, bold=True, color=BLACK)
            else:
                set_font(run, size=11, color=BLACK)

add_text(s, Inches(0.6), Inches(6.2), Inches(12.1), Inches(0.8),
         "核心判断：求解环节已有成熟工具；NL → 结构化才是 LLM 能充分发挥的地方。",
         size=16, bold=True)


# =================== SLIDE 4: NL 提取要解决的问题 ===================
s = prs.slides.add_slide(blank)
add_title(s, "3. NL → 结构化：需要解决的问题")

items = [
    "事实抽取",
    "事实忠实性",
    "格式与语义一致性",
    ("solver 直接可用", "  （含关键字段 machine_idx / duration / pre_indexes）"),
    ("solver 结果回填", "  （根据 job id + task id 回填业务字段，生成最终生产方案）"),
]
add_numbered(s, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.5),
             items, size=20, gap=14)


# =================== SLIDE 5: DSL 局限 ===================
s = prs.slides.add_slide(blank)
add_title(s, "4. DSL 方法的局限性")

dsl_items = [
    "无法处理未知数据",
    "方法复杂，需要很长的前驱准备",
    "对不合规数据无法处理，丢失后影响最终结果，且无法验证",
    "验证依赖字符匹配；JSON 结构化 / DSL 词表限制会让结果虚高",
    "没有和 ground truth 的对比",
    "从结构化数据中提取 DSL，存在「鸡生蛋」问题",
]
add_numbered(s, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.5),
             dsl_items, size=20, gap=12)


# =================== SLIDE 6: GroundTruth 问题 ===================
s = prs.slides.add_slide(blank)
add_title(s, "5. 现有 GroundTruth 的其他问题")

add_numbered(s, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.0),
             ["JSP 数据原本是标准的链式结构，被原始 GPT-4o 提取之后，"
              "duration / step 都出现了严重错漏。"],
             size=20, gap=12)

add_text(s, Inches(0.8), Inches(3.6), Inches(11.7), Inches(0.5),
         "→ 评估基线本身不可靠，需要在方法侧重建可验证链路。",
         size=18, bold=True)


# =================== SLIDE 7: 我的方法 ===================
s = prs.slides.add_slide(blank)
add_title(s, "6. 我的方法")

method_items = [
    ("三步完成信息提取：extract + format + normalize",
     ""),
    ("", "  - extract：提示词工程按 job 结构化提取；schema 反推自 JSP solver"),
    ("", "  - format：统一格式化，保证 key 匹配严格性，缩小 normalize 空间"),
    ("", "  - normalize：同义词归一化，提升字符匹配唯一性"),
    ("添加 verify 步骤，解决 extract 与 format 的正确性", ""),
    ("对提取失败或不合规的 case 标记为 bad case，方便人工介入", ""),
    ("优化 evaluation：添加 makespan 衡量 plan 实用性；改进 BLEU 反映提取正确性", ""),
    ("改进构建 JSP route sheet，让 JSP data → NL → production plan 全链路可验证", ""),
    ("可以通过简单的提示词工程更好地适配不同场景", ""),
]

# custom numbered + sub layout
tb = s.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.8))
tf = tb.text_frame; tf.word_wrap = True
tf.margin_left = tf.margin_right = Emu(0)
num = 0
first = True
for head, sub in method_items:
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    first = False
    p.alignment = PP_ALIGN.LEFT
    p.space_after = Pt(6)
    if head:
        num += 1
        r0 = p.add_run(); r0.text = f"{num}. "
        set_font(r0, size=18, bold=True)
        r1 = p.add_run(); r1.text = head
        set_font(r1, size=18)
    if sub:
        r2 = p.add_run(); r2.text = sub
        set_font(r2, size=16, color=GREY)


# =================== SLIDE 8: 核心优势 ===================
s = prs.slides.add_slide(blank)
add_title(s, "7. 核心优势")

adv = [
    "抛弃 DSL 的复杂流程，不需要预处理数据",
    "处理结构清晰：extract + format + normalize",
    "固定 schema + verify + bad case，解决 LLM 的正确性问题，允许人工介入",
    "更好的评估体系（makespan + 改进版 BLEU）",
    "对模型依赖低，4B 小模型即可胜任 —— LLM 抽取信息并 format 是非常简单的工作",
]
add_numbered(s, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.5),
             adv, size=20, gap=14)


# =================== SLIDE 9: 待讨论 ===================
s = prs.slides.add_slide(blank)
add_title(s, "8. 待讨论的问题")

qs = [
    "没有涉及提取约束的阶段 —— JSSP 本身是线性规划，是否需要独立设计这一阶段？",
    "仅提供 LLM 提取方案，是否太工程？",
    "是否需要更复杂的数据（不是 AI 生成的），来验证提取的有效性？",
]
add_numbered(s, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.5),
             qs, size=20, gap=16)


out = r"D:\FB\TASE25-AutoDSL-Constraint\note\20260702.pptx"
prs.save(out)
print(f"OK -> {out}  ({len(prs.slides)} slides)")
