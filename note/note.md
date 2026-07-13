1.dsl 本质上是做信息校验，从而提高全流程的正确性
2.llm的作用体现在帮助从自然语言变为结构化语言，方便后续计算机进行处理

1.llm信息提取+反馈，输入自然语言，用llm把自然语言转成结构化的json，然后用llm把json和自然语言的内容进行对比，找出错误的和遗漏，再转换一次，得到准确率很高的结构化json。
2.用一个大schema存储所有提取的信息，包括机器、参数和流程转信息
3.jsp求解器不需要使用机器码，全部的流转都通过key value来直接匹配
4.dsl pipeline也的operation也需要ground truth的预定义来保证一致性

machines属于是一定必要？在实际流程中是否一定有？实际生产中应该输入输入还是输出？
dsl校验过程会在很早起删掉一些信息，不具有纠错功能。

llm多次运行的一致性

新工厂没有route sheet?如何生成DSL
    鸡生蛋
    DSL无法处理没见过的数据

jsp求解器有固定输入，machine_id，duration，pre_index

对比结果和中间内容时，会出现的问题：1，llm完全提取错误 2.格式错误
当前格式错误处理：参考GT来设置prompt提取格式并校验
理论上只要归一化即可，（设置对比实验（有格式prompt和无格式prompt对比输出结果）
提取错误可以人手动修改

明确哪些信息是初始就有的（比如schema，可以通过jsp solver的输入反推并规定）

新评价指标
好模型 vs 差模型
人工修复后 vs 人工修复前vsGT
格式prompt vs 无格式prompt+lower

step3归一化会过拟合

jsp fjsp  其他类型任务的支持
当前默认线性，需要更好的匹配策略构建图/与pda对比加入一些检测

s1 s2 per step/order/instance 对比选效果
现在s1 per order，s2 per step，s4 per instance
