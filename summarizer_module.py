import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.document_loaders import TextLoader
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.text_splitter import TokenTextSplitter
    
class OpenAISummarizer:
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

    def _summarize(self, inputData, map_template, reduce_template):
        # Load documents
        loader = TextLoader(inputData, encoding='utf-8')
        docs = loader.load()

        # Create LLM Chains for map and reduce templates
        map_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(map_template))
        reduce_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(reduce_template))

        # Combine documents and reduce them
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="doc_summaries"
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=14000,
        )

        # Map and reduce documents
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )
        text_splitter = TokenTextSplitter(
            chunk_size=14000, chunk_overlap=20
        )
        split_docs = text_splitter.split_documents(docs)

        # Final result
        outputResult = map_reduce_chain.run(split_docs)
        return outputResult

    def summarizeCN(self, inputData):
        map_template = """以下是一份会议记录：\n{docs}\n根据以上会议记录, 以 Markdown 格式为会议进行严谨，严肃，的会议总结，要求按先后顺序列出所有重要信息，要求不要省略任何细节中的信息，要求标注会议发生时间和会议参加人员。\n会议报告: \n"""
        reduce_template = """以下是一份会议总结：\n{doc_summaries}\n根据以上会议记录, 作为一名会议助手，将这些内容合并，不要缩减内容，不要重复的内容，以 Markdown 格式输出，要求不要省略任何细节中的信息，要求使用中文作为输出语言。 \n会议报告:"""
        return self._summarize(inputData, map_template, reduce_template)

    def summarizeEN(self, inputData):
        map_template = """以下是一份会议记录：\n{docs}\n根据以上会议记录, 以 Markdown 格式为会议进行严谨，严肃，的会议总结，要求按先后顺序列出所有重要信息，要求不要省略任何细节中的信息，要求标注会议发生时间和会议参加人员。\n会议报告: \n"""
        reduce_template = """以下是一份会议总结：\n{doc_summaries}\n根据以上会议记录, 作为一名会议助手，将这些内容合并，不要缩减内容，不要重复的内容，以 Markdown 格式输出，要求不要省略任何细节中的信息，要求使用英文作为输出语言 \n会议报告:"""
        return self._summarize(inputData, map_template, reduce_template)

    def summarizeJDCN(self, inputData):
        map_template = """以下是一份会议或对话的记录：\n{docs}\n根据以上记录, 以 Markdown 格式为这次会议或对话，提取出一份有关Job Description的所有信息 ，要求不要省略任何细节中的信息，要求标注会议发生时间和会议参加人员，要求输出语言与会议文本中的主要语言相同。\n会议报告: \n"""
        reduce_template = """以下是一份会议或对话的大纲：\n{doc_summaries}\n根据以上记录, 作为一名会议助手，将这些内容合并，不要缩减内容，不要重复的内容，以 Markdown 格式输出一份 Job Description，要求包含：职位标题，摘要，职责和任务，技能和资格，工作地点，截止日期，联系信息等其他。要求不要省略任何细节中的信息，要求使用中文作为输出语言。 \n会议报告:"""
        return self._summarize(inputData, map_template, reduce_template)

    def summarizeJDEN(self, inputData):
        map_template = """以下是一份会议或对话的记录：\n{docs}\n根据以上记录, 以 Markdown 格式为这次会议或对话，提取出一份有关Job Description的所有信息 ，要求不要省略任何细节中的信息，要求标注会议发生时间和会议参加人员，要求输出语言与会议文本中的主要语言相同。\n会议报告: \n"""
        reduce_template = """以下是一份会议或对话的大纲：\n{doc_summaries}\n根据以上记录, 作为一名会议助手，将这些内容合并，不要缩减内容，不要重复的内容，以 Markdown 格式输出一份 Job Description，要求包含：职位标题，摘要，职责和任务，技能和资格，工作地点，截止日期，联系信息等其他。要求不要省略任何细节中的信息，要求使用英文作为输出语言。 \n会议报告:"""
        return self._summarize(inputData, map_template, reduce_template)

    def summarizeCompanyCN(self, inputData):
        map_template = """以下是一份关于一家公司的描述：\n{docs}\n根据以上内容, 提炼总结公司的基本信息包括：公司情况介绍，商业价值，技术优势，公司的位置，公司的规模，商业模式，市值/融资情况，产品，专利。\n公司介绍: \n"""
        reduce_template = """以下是一份关于一家公司的描述：：\n{doc_summaries}\n根据以上内容, 作为一名助理，将这些内容合并，不要缩减内容，给出公司信息介绍，按照格式：公司名称： {{名称}} \n 公司情况介绍： {{介绍}} \n  商业价值:{{商业价值}} \n 技术优势:{{技术优势}} \n 公司的位置:{{位置}} \n 公司的规模:{{规模}} \n 商业模式:{{商业模式}} \n 市值/融资情况:{{市值/融资}} \n 产品:{{产品}} \n 专利:{{专利}}\n以 JSON 格式输出，要求使用中文作为输出语言。 """
        return self._summarize(inputData, map_template, reduce_template)

    def summarizeCompanyEN(self, inputData):
        map_template = """以下是一份关于一家公司的描述：\n{docs}\n根据以上内容, 提炼总结公司的基本信息包括：公司情况介绍，商业价值，技术优势，公司的位置，公司的规模，商业模式，市值/融资情况，产品，专利。\n公司介绍: \n"""
        reduce_template = """以下是一份关于一家公司的描述：：\n{doc_summaries}\n根据以上内容, 作为一名助理，将这些内容合并，不要缩减内容，给出公司信息介绍，按照格式：公司名称： {{名称}} \n 公司情况介绍： {{介绍}} \n  商业价值:{{商业价值}} \n 技术优势:{{技术优势}} \n 公司的位置:{{位置}} \n 公司的规模:{{规模}} \n 商业模式:{{商业模式}} \n 市值/融资情况:{{市值/融资}} \n 产品:{{产品}} \n 专利:{{专利}}\n以 JSON 格式输出，要求使用英文作为输出语言。 """
        return self._summarize(inputData, map_template, reduce_template)