from mcp.server.fastmcp import FastMCP
from tools import filter_data_with_df, get_holidays_in_months, get_weather, get_sns_url, load_df
from tools import ToolOutput


DF = load_df()

mcp = FastMCP()

def filter_data(name: str) -> ToolOutput:
    """사용자가 입력한 가맹점 이름을 기준으로 데이터를 필터링합니다."""
    return filter_data_with_df(df=DF, name=name)

mcp.tool(
    description="""
    사용자가 입력한 가맹점 이름을 기준으로 데이터를 필터링해 반환하는 함수입니다.
    """,
    structured_output=True
)(filter_data)

mcp.tool(
    description="""
    오늘 날짜를 기준으로 가까운 기념일을 검색하여 문자열 리스트로 반환하는 함수입니다. 
    """,
    structured_output= True
)(get_holidays_in_months)

mcp.tool(
    description="""
    오늘 날짜를 기준으로 5일 간의 날씨, 온도를 알려주는 함수입니다.
    """,
    structured_output= True
)(get_weather)

mcp.tool(
    description="""
    주로 사용하는 SNS의 정확한 URL을 반환하는 함수입니다.
    """,
    structured_output= True
)(get_sns_url)


if __name__ == "__main__":
    mcp.run(transport="stdio")

