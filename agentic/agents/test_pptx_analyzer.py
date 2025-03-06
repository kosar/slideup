from agentic.agents.pptx_analyzer import PptxAnalyzerAgent

agent = PptxAnalyzerAgent()
result = agent.process({'pptx_path': './test.pptx'})
print(result.get('analysis_report'))
