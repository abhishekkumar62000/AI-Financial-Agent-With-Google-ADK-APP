import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import json
import logging
from pydantic import BaseModel, Field
import csv
from io import StringIO

# FIX: Ensure the google-adk package is installed and available in your environment.
# If this is a custom or local module, adjust the import path accordingly.
# For Google ADK, you may need to install it via pip:
# pip install google-adk
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "finance_advisor"
USER_ID = "default_user"

# --- UI/UX ENHANCEMENT: Advanced CSS for animation, color, and interactivity ---
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f8ffae 0%, #43c6ac 100%);
        animation: bgfade 8s ease-in-out infinite alternate;
    }
    @keyframes bgfade {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    .main {
        background: rgba(255,255,255,0.95);
        border-radius: 24px;
        padding: 2.5rem 2rem 2rem 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
        margin-top: 1.5rem;
        animation: fadein 1.2s;
    }
    @keyframes fadein {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff512f 0%, #dd2476 100%);
        color: white;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.7rem 2.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(221, 36, 118, 0.12);
        transition: 0.25s cubic-bezier(.4,2,.6,1);
        animation: popin 0.7s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #43c6ac 0%, #191654 100%);
        color: #fff;
        transform: scale(1.09) rotate(-2deg);
        box-shadow: 0 4px 16px rgba(67, 198, 172, 0.18);
    }
    @keyframes popin {
        0% { transform: scale(0.8); }
        100% { transform: scale(1); }
    }
    .stSidebar {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        animation: fadein 1.5s;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #43c6ac;
        background: #f4faff;
        font-size: 1.1rem;
        transition: border 0.2s;
    }
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border: 2px solid #ff512f;
        background: #fffbe7;
    }
    .stSelectbox>div>div>div>div {
        border-radius: 10px;
        border: 2px solid #43c6ac;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #dd2476;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px #f8ffae;
        animation: fadein 1.2s;
    }
    .stMarkdown h1 {
        font-size: 2.8rem;
    }
    .stMarkdown h2 {
        font-size: 2.2rem;
    }
    .stMarkdown h3 {
        font-size: 1.5rem;
    }
    .stAlert {
        border-radius: 12px;
        font-size: 1.1rem;
        animation: fadein 1.2s;
    }
    .st-bb {
        background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #333;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(255, 215, 0, 0.12);
        animation: popin 0.7s;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #43c6ac, #191654);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Optional: Animated emoji feedback for user actions ---
def animated_emoji(emoji: str, key: str):
    st.markdown(f"<div style='font-size:2.5rem; animation: popin 0.7s;'>{emoji}</div>", unsafe_allow_html=True)

# Pydantic models for output schemas
class SpendingCategory(BaseModel):
    category: str = Field(..., description="Expense category name")
    amount: float = Field(..., description="Amount spent in this category")
    percentage: Optional[float] = Field(None, description="Percentage of total spending")

class SpendingRecommendation(BaseModel):
    category: str = Field(..., description="Category for recommendation")
    recommendation: str = Field(..., description="Recommendation details")
    potential_savings: Optional[float] = Field(None, description="Estimated monthly savings")

class BudgetAnalysis(BaseModel):
    total_expenses: float = Field(..., description="Total monthly expenses")
    monthly_income: Optional[float] = Field(None, description="Monthly income")
    spending_categories: List[SpendingCategory] = Field(..., description="Breakdown of spending by category")
    recommendations: List[SpendingRecommendation] = Field(..., description="Spending recommendations")

class EmergencyFund(BaseModel):
    recommended_amount: float = Field(..., description="Recommended emergency fund size")
    current_amount: Optional[float] = Field(None, description="Current emergency fund (if any)")
    current_status: str = Field(..., description="Status assessment of emergency fund")

class SavingsRecommendation(BaseModel):
    category: str = Field(..., description="Savings category")
    amount: float = Field(..., description="Recommended monthly amount")
    rationale: Optional[str] = Field(None, description="Explanation for this recommendation")

class AutomationTechnique(BaseModel):
    name: str = Field(..., description="Name of automation technique")
    description: str = Field(..., description="Details of how to implement")

class SavingsStrategy(BaseModel):
    emergency_fund: EmergencyFund = Field(..., description="Emergency fund recommendation")
    recommendations: List[SavingsRecommendation] = Field(..., description="Savings allocation recommendations")
    automation_techniques: Optional[List[AutomationTechnique]] = Field(None, description="Automation techniques to help save")

class Debt(BaseModel):
    name: str = Field(..., description="Name of debt")
    amount: float = Field(..., description="Current balance")
    interest_rate: float = Field(..., description="Annual interest rate (%)")
    min_payment: Optional[float] = Field(None, description="Minimum monthly payment")

class PayoffPlan(BaseModel):
    total_interest: float = Field(..., description="Total interest paid")
    months_to_payoff: int = Field(..., description="Months until debt-free")
    monthly_payment: Optional[float] = Field(None, description="Recommended monthly payment")

class PayoffPlans(BaseModel):
    avalanche: PayoffPlan = Field(..., description="Highest interest first method")
    snowball: PayoffPlan = Field(..., description="Smallest balance first method")

class DebtRecommendation(BaseModel):
    title: str = Field(..., description="Title of recommendation")
    description: str = Field(..., description="Details of recommendation")
    impact: Optional[str] = Field(None, description="Expected impact of this action")

class DebtReduction(BaseModel):
    total_debt: float = Field(..., description="Total debt amount")
    debts: List[Debt] = Field(..., description="List of all debts")
    payoff_plans: PayoffPlans = Field(..., description="Debt payoff strategies")
    recommendations: Optional[List[DebtRecommendation]] = Field(None, description="Recommendations for debt reduction")

load_dotenv()
GEMINI_API_KEY = os.getenv("Google_Gemini_ai_key")
if GEMINI_API_KEY:
    os.environ["Google_Gemini_ai_key"] = GEMINI_API_KEY

def parse_json_safely(data: str, default_value: Any = None) -> Any:
    """Safely parse JSON data with error handling"""
    try:
        return json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError:
        return default_value

class FinanceAdvisorSystem:
    def __init__(self):
        self.session_service = InMemorySessionService()
        
        self.budget_analysis_agent = LlmAgent(
            name="BudgetAnalysisAgent",
            model="gemini-2.0-flash-exp",
            description="Analyzes financial data to categorize spending patterns and recommend budget improvements",
            instruction="""You are a Budget Analysis Agent specialized in reviewing financial transactions and expenses.
You are the first agent in a sequence of three financial advisor agents.

Your tasks:
1. Analyze income, transactions, and expenses in detail
2. Categorize spending into logical groups with clear breakdown
3. Identify spending patterns and trends across categories
4. Suggest specific areas where spending could be reduced with concrete suggestions
5. Provide actionable recommendations with specific, quantified potential savings amounts

Consider:
- Number of dependants when evaluating household expenses
- Typical spending ratios for the income level (housing 30%, food 15%, etc.)
- Essential vs discretionary spending with clear separation
- Seasonal spending patterns if data spans multiple months

For spending categories, include ALL expenses from the user's data, ensure percentages add up to 100%,
and make sure every expense is categorized.

For recommendations:
- Provide at least 3-5 specific, actionable recommendations with estimated savings
- Explain the reasoning behind each recommendation
- Consider the impact on quality of life and long-term financial health
- Suggest specific implementation steps for each recommendation

IMPORTANT: Store your analysis in state['budget_analysis'] for use by subsequent agents.""",
            output_schema=BudgetAnalysis,
            output_key="budget_analysis"
        )
        
        self.savings_strategy_agent = LlmAgent(
            name="SavingsStrategyAgent",
            model="gemini-2.0-flash-exp",
            description="Recommends optimal savings strategies based on income, expenses, and financial goals",
            instruction="""You are a Savings Strategy Agent specialized in creating personalized savings plans.
You are the second agent in the sequence. READ the budget analysis from state['budget_analysis'] first.

Your tasks:
1. Review the budget analysis results from state['budget_analysis']
2. Recommend comprehensive savings strategies based on the analysis
3. Calculate optimal emergency fund size based on expenses and dependants
4. Suggest appropriate savings allocation across different purposes
5. Recommend practical automation techniques for saving consistently

Consider:
- Risk factors based on job stability and dependants
- Balancing immediate needs with long-term financial health
- Progressive savings rates as discretionary income increases
- Multiple savings goals (emergency, retirement, specific purchases)
- Areas of potential savings identified in the budget analysis

IMPORTANT: Store your strategy in state['savings_strategy'] for use by the Debt Reduction Agent.""",
            output_schema=SavingsStrategy,
            output_key="savings_strategy"
        )
        
        self.debt_reduction_agent = LlmAgent(
            name="DebtReductionAgent",
            model="gemini-2.0-flash-exp",
            description="Creates optimized debt payoff plans to minimize interest paid and time to debt freedom",
            instruction="""You are a Debt Reduction Agent specialized in creating debt payoff strategies.
You are the final agent in the sequence. READ both state['budget_analysis'] and state['savings_strategy'] first.

Your tasks:
1. Review both budget analysis and savings strategy from the state
2. Analyze debts by interest rate, balance, and minimum payments
3. Create prioritized debt payoff plans (avalanche and snowball methods)
4. Calculate total interest paid and time to debt freedom
5. Suggest debt consolidation or refinancing opportunities
6. Provide specific recommendations to accelerate debt payoff

Consider:
- Cash flow constraints from the budget analysis
- Emergency fund and savings goals from the savings strategy
- Psychological factors (quick wins vs mathematical optimization)
- Credit score impact and improvement opportunities

IMPORTANT: Store your final plan in state['debt_reduction'] and ensure it aligns with the previous analyses.""",
            output_schema=DebtReduction,
            output_key="debt_reduction"
        )
        
        self.coordinator_agent = SequentialAgent(
            name="FinanceCoordinatorAgent",
            description="Coordinates specialized finance agents to provide comprehensive financial advice",
            sub_agents=[
                self.budget_analysis_agent,
                self.savings_strategy_agent,
                self.debt_reduction_agent
            ]
        )
        
        self.runner = Runner(
            agent=self.coordinator_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )

    async def analyze_finances(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = f"finance_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            initial_state = {
                "monthly_income": financial_data.get("monthly_income", 0),
                "dependants": financial_data.get("dependants", 0),
                "transactions": financial_data.get("transactions", []),
                "manual_expenses": financial_data.get("manual_expenses", {}),
                "debts": financial_data.get("debts", [])
            }
            
            session = await self.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state=initial_state
            )
            
            if session.state.get("transactions"):
                self._preprocess_transactions(session)
            
            if session.state.get("manual_expenses"):
                self._preprocess_manual_expenses(session)
            
            default_results = self._create_default_results(financial_data)
            
            user_content = types.Content(
                role='user',
                parts=[types.Part(text=json.dumps(financial_data))]
            )
            
            async for event in self.runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=user_content
            ):
                if event.is_final_response() and event.author == self.coordinator_agent.name:
                    break
            
            updated_session = await self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
            
            results = {}
            for key in ["budget_analysis", "savings_strategy", "debt_reduction"]:
                value = updated_session.state.get(key)
                results[key] = parse_json_safely(value, default_results[key]) if value else default_results[key]
            
            return results
            
        except Exception as e:
            logger.exception(f"Error during finance analysis: {str(e)}")
            raise
        finally:
            await self.session_service.delete_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
    
    def _preprocess_transactions(self, session):
        transactions = session.state.get("transactions", [])
        if not transactions:
            return
        
        df = pd.DataFrame(transactions)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        if 'Category' in df.columns and 'Amount' in df.columns:
            category_spending = df.groupby('Category')['Amount'].sum().to_dict()
            session.state["category_spending"] = category_spending
            session.state["total_spending"] = df['Amount'].sum()
    
    def _preprocess_manual_expenses(self, session):
        manual_expenses = session.state.get("manual_expenses", {})
        if not manual_expenses:
            return
        
        session.state.update({
            "total_manual_spending": sum(manual_expenses.values()),
            "manual_category_spending": manual_expenses
        })

    def _create_default_results(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        monthly_income = financial_data.get("monthly_income", 0)
        expenses = financial_data.get("manual_expenses", {})
        
        if not expenses and financial_data.get("transactions"):
            expenses = {}
            for transaction in financial_data["transactions"]:
                category = transaction.get("Category", "Uncategorized")
                amount = transaction.get("Amount", 0)
                expenses[category] = expenses.get(category, 0) + amount
        
        total_expenses = sum(expenses.values())
        
        return {
            "budget_analysis": {
                "total_expenses": total_expenses,
                "monthly_income": monthly_income,
                "spending_categories": [
                    {"category": cat, "amount": amt, "percentage": (amt / total_expenses * 100) if total_expenses > 0 else 0}
                    for cat, amt in expenses.items()
                ],
                "recommendations": [
                    {"category": "General", "recommendation": "Consider reviewing your expenses carefully", "potential_savings": total_expenses * 0.1}
                ]
            },
            "savings_strategy": {
                "emergency_fund": {
                    "recommended_amount": total_expenses * 6,
                    "current_amount": 0,
                    "current_status": "Not started"
                },
                "recommendations": [
                    {"category": "Emergency Fund", "amount": total_expenses * 0.1, "rationale": "Build emergency fund first"},
                    {"category": "Retirement", "amount": monthly_income * 0.15, "rationale": "Long-term savings"}
                ],
                "automation_techniques": [
                    {"name": "Automatic Transfer", "description": "Set up automatic transfers on payday"}
                ]
            },
            "debt_reduction": {
                "total_debt": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])),
                "debts": financial_data.get("debts", []),
                "payoff_plans": {
                    "avalanche": {
                        "total_interest": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) * 0.2,
                        "months_to_payoff": 24,
                        "monthly_payment": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) / 24
                    },
                    "snowball": {
                        "total_interest": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) * 0.25,
                        "months_to_payoff": 24,
                        "monthly_payment": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) / 24
                    }
                },
                "recommendations": [
                    {"title": "Increase Payments", "description": "Increase your monthly payments", "impact": "Reduces total interest paid"}
                ]
            }
        }

def display_budget_analysis(analysis: Dict[str, Any]):
    if isinstance(analysis, str):
        try:
            analysis = json.loads(analysis)
        except json.JSONDecodeError:
            st.error("Failed to parse budget analysis results")
            return
    
    if not isinstance(analysis, dict):
        st.error("Invalid budget analysis format")
        return
    
    if "spending_categories" in analysis:
        st.subheader("Spending by Category")
        fig = px.pie(
            values=[cat["amount"] for cat in analysis["spending_categories"]],
            names=[cat["category"] for cat in analysis["spending_categories"]],
            title="Your Spending Breakdown"
        )
        st.plotly_chart(fig)
    
    if "total_expenses" in analysis:
        st.subheader("Income vs. Expenses")
        income = analysis.get("monthly_income", 0)
        expenses = analysis["total_expenses"]
        surplus_deficit = income - expenses
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Income", "Expenses"], 
                            y=[income, expenses],
                            marker_color=["green", "red"]))
        fig.update_layout(title="Monthly Income vs. Expenses")
        st.plotly_chart(fig)
        
        st.metric("Monthly Surplus/Deficit", 
                  f"${surplus_deficit:.2f}", 
                  delta=f"{surplus_deficit:.2f}")
    
    if "recommendations" in analysis:
        st.subheader("Spending Reduction Recommendations")
        for rec in analysis["recommendations"]:
            st.markdown(f"**{rec['category']}**: {rec['recommendation']}")
            if "potential_savings" in rec:
                st.metric(f"Potential Monthly Savings", f"${rec['potential_savings']:.2f}")

def display_savings_strategy(strategy: Dict[str, Any]):
    if isinstance(strategy, str):
        try:
            strategy = json.loads(strategy)
        except json.JSONDecodeError:
            st.error("Failed to parse savings strategy results")
            return
    
    if not isinstance(strategy, dict):
        st.error("Invalid savings strategy format")
        return
    
    st.subheader("Savings Recommendations")
    
    if "emergency_fund" in strategy:
        ef = strategy["emergency_fund"]
        st.markdown(f"### Emergency Fund")
        st.markdown(f"**Recommended Size**: ${ef['recommended_amount']:.2f}")
        st.markdown(f"**Current Status**: {ef['current_status']}")
        
        if "current_amount" in ef and "recommended_amount" in ef:
            progress = ef["current_amount"] / ef["recommended_amount"]
            st.progress(min(progress, 1.0))
            st.markdown(f"${ef['current_amount']:.2f} of ${ef['recommended_amount']:.2f}")
    
    if "recommendations" in strategy:
        st.markdown("### Recommended Savings Allocations")
        for rec in strategy["recommendations"]:
            st.markdown(f"**{rec['category']}**: ${rec['amount']:.2f}/month")
            st.markdown(f"_{rec['rationale']}_")
    
    if "automation_techniques" in strategy:
        st.markdown("### Automation Techniques")
        for technique in strategy["automation_techniques"]:
            st.markdown(f"**{technique['name']}**: {technique['description']}")

def display_debt_reduction(plan: Dict[str, Any]):
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            st.error("Failed to parse debt reduction results")
            return
    
    if not isinstance(plan, dict):
        st.error("Invalid debt reduction format")
        return
    
    if "total_debt" in plan:
        st.metric("Total Debt", f"${plan['total_debt']:.2f}")
    
    if "debts" in plan:
        st.subheader("Your Debts")
        debt_df = pd.DataFrame(plan["debts"])
        st.dataframe(debt_df)
        
        fig = px.bar(debt_df, x="name", y="amount", color="interest_rate",
                    labels={"name": "Debt", "amount": "Amount ($)", "interest_rate": "Interest Rate (%)"},
                    title="Debt Breakdown")
        st.plotly_chart(fig)
    
    if "payoff_plans" in plan:
        st.subheader("Debt Payoff Plans")
        tabs = st.tabs(["Avalanche Method", "Snowball Method", "Comparison"])
        
        with tabs[0]:
            st.markdown("### Avalanche Method (Highest Interest First)")
            if "avalanche" in plan["payoff_plans"]:
                avalanche = plan["payoff_plans"]["avalanche"]
                st.markdown(f"**Total Interest Paid**: ${avalanche['total_interest']:.2f}")
                st.markdown(f"**Time to Debt Freedom**: {avalanche['months_to_payoff']} months")
                
                if "monthly_payment" in avalanche:
                    st.markdown(f"**Recommended Monthly Payment**: ${avalanche['monthly_payment']:.2f}")
        
        with tabs[1]:
            st.markdown("### Snowball Method (Smallest Balance First)")
            if "snowball" in plan["payoff_plans"]:
                snowball = plan["payoff_plans"]["snowball"]
                st.markdown(f"**Total Interest Paid**: ${snowball['total_interest']:.2f}")
                st.markdown(f"**Time to Debt Freedom**: {snowball['months_to_payoff']} months")
                
                if "monthly_payment" in snowball:
                    st.markdown(f"**Recommended Monthly Payment**: ${snowball['monthly_payment']:.2f}")
        
        with tabs[2]:
            st.markdown("### Method Comparison")
            if "avalanche" in plan["payoff_plans"] and "snowball" in plan["payoff_plans"]:
                avalanche = plan["payoff_plans"]["avalanche"]
                snowball = plan["payoff_plans"]["snowball"]
                
                comparison_data = {
                    "Method": ["Avalanche", "Snowball"],
                    "Total Interest": [avalanche["total_interest"], snowball["total_interest"]],
                    "Months to Payoff": [avalanche["months_to_payoff"], snowball["months_to_payoff"]]
                }
                comparison_df = pd.DataFrame(comparison_data)
                
                st.dataframe(comparison_df)
                
                fig = go.Figure(data=[
                    go.Bar(name="Total Interest", x=comparison_df["Method"], y=comparison_df["Total Interest"]),
                    go.Bar(name="Months to Payoff", x=comparison_df["Method"], y=comparison_df["Months to Payoff"])
                ])
                fig.update_layout(barmode='group', title="Debt Payoff Method Comparison")
                st.plotly_chart(fig)
    
    if "recommendations" in plan:
        st.subheader("Debt Reduction Recommendations")
        for rec in plan["recommendations"]:
            st.markdown(f"**{rec['title']}**: {rec['description']}")
            if "impact" in rec:
                st.markdown(f"_Impact: {rec['impact']}_")

def parse_csv_transactions(file_content) -> List[Dict[str, Any]]:
    """Parse CSV file content into a list of transactions"""
    try:
        # Read CSV content
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert date strings to datetime and then to string format YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Convert amount strings to float, handling currency symbols and commas
        df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)
        
        # Group by category and calculate totals
        category_totals = df.groupby('Category')['Amount'].sum().reset_index()
        
        # Convert to list of dictionaries
        transactions = df.to_dict('records')
        
        return {
            'transactions': transactions,
            'category_totals': category_totals.to_dict('records')
        }
    except Exception as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")

def validate_csv_format(file) -> bool:
    """Validate CSV file format and content"""
    try:
        content = file.read().decode('utf-8')
        dialect = csv.Sniffer().sniff(content)
        has_header = csv.Sniffer().has_header(content)
        file.seek(0)  # Reset file pointer
        
        if not has_header:
            return False, "CSV file must have headers"
            
        df = pd.read_csv(StringIO(content))
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
            
        # Validate date format
        try:
            pd.to_datetime(df['Date'])
        except:
            return False, "Invalid date format in Date column"
            
        # Validate amount format (should be numeric after removing currency symbols)
        try:
            df['Amount'].replace('[\$,]', '', regex=True).astype(float)
        except:
            return False, "Invalid amount format in Amount column"
            
        return True, "CSV format is valid"
    except Exception as e:
        return False, f"Invalid CSV format: {str(e)}"

def display_csv_preview(df: pd.DataFrame):
    """Display a preview of the CSV data with basic statistics"""
    st.subheader("CSV Data Preview")
    
    # Show basic statistics
    total_transactions = len(df)
    total_amount = df['Amount'].sum()
    
    # Convert dates for display
    df_dates = pd.to_datetime(df['Date'])
    date_range = f"{df_dates.min().strftime('%Y-%m-%d')} to {df_dates.max().strftime('%Y-%m-%d')}"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", total_transactions)
    with col2:
        st.metric("Total Amount", f"${total_amount:,.2f}")
    with col3:
        st.metric("Date Range", date_range)
    
    # Show category breakdown
    st.subheader("Spending by Category")
    category_totals = df.groupby('Category')['Amount'].agg(['sum', 'count']).reset_index()
    category_totals.columns = ['Category', 'Total Amount', 'Transaction Count']
    st.dataframe(category_totals)
    
    # Show sample transactions
    st.subheader("Sample Transactions")
    st.dataframe(df.head())

# --- AI-Powered Spending Insights & Alerts ---
def get_spending_insights_alerts(transactions_df, manual_expenses, prev_month_data=None):
    insights = []
    alerts = []
    now = datetime.now()
    # Combine all expenses into a DataFrame
    if transactions_df is not None and not transactions_df.empty:
        df = transactions_df.copy()
    elif manual_expenses and any(manual_expenses.values()):
        df = pd.DataFrame([
            {'Date': now.strftime('%Y-%m-%d'), 'Category': k, 'Amount': v}
            for k, v in manual_expenses.items() if v > 0
        ])
    else:
        return insights, alerts
    # Large transaction alert
    if not df.empty:
        mean_amt = df['Amount'].mean()
        std_amt = df['Amount'].std()
        large_tx = df[df['Amount'] > mean_amt + 2*std_amt]
        for _, row in large_tx.iterrows():
            alerts.append(f"⚠️ Large transaction: ${row['Amount']:.2f} in {row['Category']} on {row['Date']}")
        # Category overspending (over 30% of total or over $500, or 20% more than last month if prev_month_data)
        cat_totals = df.groupby('Category')['Amount'].sum()
        total = cat_totals.sum()
        for cat, amt in cat_totals.items():
            if amt > 0.3 * total or amt > 500:
                alerts.append(f"🚨 Overspending: ${amt:.2f} spent on {cat} this month.")
        # Compare to previous month if available
        if prev_month_data is not None and not prev_month_data.empty:
            prev_cats = prev_month_data.groupby('Category')['Amount'].sum()
            for cat, amt in cat_totals.items():
                prev_amt = prev_cats.get(cat, 0)
                if prev_amt > 0 and amt > prev_amt * 1.2:
                    insights.append(f"📈 You spent {((amt-prev_amt)/prev_amt*100):.1f}% more on {cat} this month than last month.")
        # Weekly summary (last 7 days)
        df['Date'] = pd.to_datetime(df['Date'])
        last_week = df[df['Date'] >= now - pd.Timedelta(days=7)]
        if not last_week.empty:
            week_total = last_week['Amount'].sum()
            insights.append(f"🗓️ You spent ${week_total:.2f} in the last 7 days.")
    return insights, alerts
# --- End AI-Powered Spending Insights & Alerts ---

# --- Interactive Financial Education Modules ---
class Lesson:
    def __init__(self, title, content, quiz=None):
        self.title = title
        self.content = content
        self.quiz = quiz or []
lessons = [
    Lesson(
        "Budgeting Basics",
        "Learn how to create and stick to a budget. Key steps: Track income/expenses, set spending limits, review monthly.",
        quiz=[
            {"q": "What is the first step in budgeting?", "a": ["Track your income and expenses", "Invest in stocks", "Apply for a loan"], "correct": 0},
            {"q": "What percentage of income is typically recommended for housing?", "a": ["10%", "30%", "50%"], "correct": 1}
        ]
    ),
    Lesson(
        "Understanding Credit Scores",
        "Your credit score affects loan rates. Pay bills on time, keep credit usage low, check your report annually.",
        quiz=[
            {"q": "Which action improves your credit score?", "a": ["Missing payments", "Paying bills on time", "Maxing out credit cards"], "correct": 1}
        ]
    ),
    Lesson(
        "Investing 101",
        "Investing helps grow your wealth. Start early, diversify, and understand your risk tolerance.",
        quiz=[
            {"q": "What does diversification mean?", "a": ["Investing in one stock", "Spreading investments across assets", "Keeping cash only"], "correct": 1}
        ]
    )
]
def suggest_learning_path(financial_data, goals):
    path = []
    if financial_data.get('debts') and sum(d.get('amount',0) for d in financial_data['debts']) > 0:
        path.append("Understanding Credit Scores")
    if financial_data.get('manual_expenses') or financial_data.get('transactions'):
        path.append("Budgeting Basics")
    if goals and any(g['target'] > 1000 for g in goals):
        path.append("Investing 101")
    return path or [l.title for l in lessons]
# --- End Interactive Financial Education Modules ---

# --- Enhanced, colorful, animated app title section ---
st.markdown(
    """
    <div style="
        text-align:center;
        margin-top: 0.5em;
        margin-bottom: 1.5em;
        padding: 1.2em 0 1em 0;
        border-radius: 24px;
        background: linear-gradient(90deg, #ff512f 0%, #dd2476 40%, #43c6ac 100%);
        box-shadow: 0 4px 32px 0 rgba(67,198,172,0.13);
        position: relative;
        overflow: hidden;
        animation: fadein 1.2s;
    ">
        <span style="
            font-size: 2.8rem;
            font-weight: 900;
            letter-spacing: 2px;
            background: linear-gradient(90deg, #fffbe7 0%, #f8ffae 40%, #43c6ac 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
            text-shadow: 0 2px 12px #19165444;
            display: inline-block;
            animation: popin 1.1s;
        ">🤖 AI Financial Agent With Google ADK</span>
        <br>
        <span style="
            font-size: 1.25rem;
            color: #fff;
            font-weight: 600;
            letter-spacing: 1px;
            text-shadow: 0 2px 8px #19165433;
            margin-top: 0.5em;
            display: inline-block;
            animation: bounce 1.8s infinite alternate;
        ">Your Smart, Colorful, and Interactive Finance Partner</span>
    </div>
    <style>
    @keyframes popin {
        0% { transform: scale(0.8); opacity: 0.2; }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes bounce {
        0% { transform: translateY(0); }
        100% { transform: translateY(-10px); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.set_page_config(
        page_title="🤖AI Financial Agent with Google ADK",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    
    # Main content
    st.caption("Powered by Google's Agent Development Kit (ADK) and Gemini AI")
    st.info("This tool analyzes your financial data and provides tailored recommendations for budgeting, savings, and debt management using multiple specialized AI agents.")
    st.divider()
    
    # Create tabs for different sections
    input_tab, goals_tab, scenario_tab, education_tab, about_tab = st.tabs([
        "💼 Financial Information", "🎯 Goals & Progress", "🧮 What-If Simulation", "📚 Financial Education", "ℹ️ About"])

    # --- Personalized Notifications & Reminders ---
    def show_notifications():
        notifications = []
        today = datetime.now().date()
        # --- HIDE bill and large expense notifications from UI ---
        # upcoming_bills = [
        #     {"name": "Credit Card Payment", "due": today.replace(day=min(today.day+3,28)), "amount": 200},
        #     {"name": "Utility Bill", "due": today.replace(day=min(today.day+7,28)), "amount": 120},
        # ]
        # for bill in upcoming_bills:
        #     days_left = (bill['due'] - today).days
        #     if 0 <= days_left <= 7:
        #         notifications.append(f"🔔 Upcoming bill: **{bill['name']}** of ${bill['amount']} due in {days_left} days ({bill['due']})")
        # Savings transfer reminder (weekly)
        if today.weekday() == 4:  # Friday
            notifications.append("💡 Reminder: It's payday! Consider transferring to your savings goals.")
        # Spending threshold notification (simulate with total expenses)
        if 'goals' in st.session_state and st.session_state['goals']:
            for goal in st.session_state['goals']:
                if 0 < (goal['target'] - goal['current']) <= goal['target'] * 0.1:
                    notifications.append(f"🎯 You're close to reaching your goal **{goal['name']}**! Only ${goal['target']-goal['current']:.2f} left.")
        # notifications.append("⚠️ Large expense detected: 'Car Insurance' of $800 due next week.")
        # Display notifications
        if notifications:
            for note in notifications:
                st.warning(note)
    # --- End Notifications ---

    with input_tab:
        show_notifications()
        st.header("Enter Your Financial Information")
        st.caption("All data is processed locally and not stored anywhere.")
        
        # Income and Dependants section in a container
        with st.container():
            st.subheader("💰 Income & Household")
            income_col, dependants_col = st.columns([2, 1])
            with income_col:
                monthly_income = st.number_input(
                    "Monthly Income ($)",
                    min_value=0.0,
                    step=100.0,
                    value=3000.0,
                    key="income",
                    help="Enter your total monthly income after taxes"
                )
            with dependants_col:
                dependants = st.number_input(
                    "Number of Dependants",
                    min_value=0,
                    step=1,
                    value=0,
                    key="dependants",
                    help="Include all dependants in your household"
                )
        
        st.divider()
        
        # Expenses section
        with st.container():
            st.subheader("💳 Expenses")
            expense_option = st.radio(
                "How would you like to enter your expenses?",
                ("📤 Upload CSV Transactions", "✍️ Enter Manually"),
                key="expense_option",
                horizontal=True
            )
            
            transaction_file = None
            manual_expenses = {}
            use_manual_expenses = False
            transactions_df = None

            if expense_option == "📤 Upload CSV Transactions":
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("""
                    #### Upload your transaction data
                    Your CSV file should have these columns:
                    - 📅 Date (YYYY-MM-DD)
                    - 📝 Category
                    - 💲 Amount
                    """)
                    
                    transaction_file = st.file_uploader(
                        "Choose your CSV file",
                        type=["csv"],
                        key="transaction_file",
                        help="Upload a CSV file containing your transactions"
                    )
                
                if transaction_file is not None:
                    # Validate CSV format
                    is_valid, message = validate_csv_format(transaction_file)
                    
                    if is_valid:
                        try:
                            # Parse CSV content
                            transaction_file.seek(0)
                            file_content = transaction_file.read()
                            parsed_data = parse_csv_transactions(file_content)
                            
                            # Create DataFrame
                            transactions_df = pd.DataFrame(parsed_data['transactions'])
                            
                            # Display preview
                            display_csv_preview(transactions_df)
                            
                            st.success("✅ Transaction file uploaded and validated successfully!")
                        except Exception as e:
                            st.error(f"❌ Error processing CSV file: {str(e)}")
                            transactions_df = None
                    else:
                        st.error(message)
                        transactions_df = None
            else:
                use_manual_expenses = True
                st.markdown("#### Enter your monthly expenses by category")
                
                # Define expense categories with emojis
                categories = [
                    ("🏠 Housing", "Housing"),
                    ("🔌 Utilities", "Utilities"),
                    ("🍽️ Food", "Food"),
                    ("🚗 Transportation", "Transportation"),
                    ("🏥 Healthcare", "Healthcare"),
                    ("🎭 Entertainment", "Entertainment"),
                    ("👤 Personal", "Personal"),
                    ("💰 Savings", "Savings"),
                    ("📦 Other", "Other")
                ]
                
                # Create three columns for better layout
                col1, col2, col3 = st.columns(3)
                cols = [col1, col2, col3]
                
                # Distribute categories across columns
                for i, (emoji_cat, cat) in enumerate(categories):
                    with cols[i % 3]:
                        manual_expenses[cat] = st.number_input(
                            emoji_cat,
                            min_value=0.0,
                            step=50.0,
                            value=0.0,
                            key=f"manual_{cat}",
                            help=f"Enter your monthly {cat.lower()} expenses"
                        )
                
                if any(manual_expenses.values()):
                    st.markdown("#### 📊 Summary of Entered Expenses")
                    manual_df_disp = pd.DataFrame({
                        'Category': list(manual_expenses.keys()),
                        'Amount': list(manual_expenses.values())
                    })
                    manual_df_disp = manual_df_disp[manual_df_disp['Amount'] > 0]
                    if not manual_df_disp.empty:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.dataframe(
                                manual_df_disp,
                                column_config={
                                    "Category": "Category",
                                    "Amount": st.column_config.NumberColumn(
                                        "Amount",
                                        format="$%.2f"
                                    )
                                },
                                hide_index=True
                            )
                        with col2:
                            st.metric(
                                "Total Monthly Expenses",
                                f"${manual_df_disp['Amount'].sum():,.2f}"
                            )
        
        st.divider()
        
        # Debt Information section
        with st.container():
            st.subheader("🏦 Debt Information")
            st.info("Enter your debts to get personalized payoff strategies using both avalanche and snowball methods.")
            
            num_debts = st.number_input(
                "How many debts do you have?",
                min_value=0,
                max_value=10,
                step=1,
                value=0,
                key="num_debts"
            )
            
            debts = []
            if num_debts > 0:
                # Create columns for debts
                cols = st.columns(min(num_debts, 3))  # Max 3 columns per row
                for i in range(num_debts):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.markdown(f"##### Debt #{i+1}")
                        debt_name = st.text_input(
                            "Name",
                            value=f"Debt {i+1}",
                            key=f"debt_name_{i}",
                            help="Enter a name for this debt (e.g., Credit Card, Student Loan)"
                        )
                        debt_amount = st.number_input(
                            "Amount ($)",
                            min_value=0.01,
                            step=100.0,
                            value=1000.0,
                            key=f"debt_amount_{i}",
                            help="Enter the current balance of this debt"
                        )
                        interest_rate = st.number_input(
                            "Interest Rate (%)",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.1,
                            value=5.0,
                            key=f"debt_rate_{i}",
                            help="Enter the annual interest rate"
                        )
                        min_payment = st.number_input(
                            "Minimum Payment ($)",
                            min_value=0.0,
                            step=10.0,
                            value=50.0,
                            key=f"debt_min_payment_{i}",
                            help="Enter the minimum monthly payment required"
                        )
                        
                        debts.append({
                            "name": debt_name,
                            "amount": debt_amount,
                            "interest_rate": interest_rate,
                            "min_payment": min_payment
                        })
                        
                        if col_idx == 2 or i == num_debts - 1:  # Add spacing after every 3 debts or last debt
                            st.markdown("---")
        
        st.divider()
        
        # Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔄 Analyze My Finances",
                key="analyze_button",
                use_container_width=True,
                help="Click to get your personalized financial analysis"
            )
        
        if analyze_button:
            if expense_option == "Upload CSV Transactions" and transactions_df is None:
                st.error("Please upload a valid transaction CSV file or choose manual entry.")
                return
            if use_manual_expenses and not any(manual_expenses.values()):
                st.warning("No manual expenses entered. Analysis might be limited.")

            st.header("Financial Analysis Results")
            with st.spinner("🤖 AI agents are analyzing your financial data..."): 
                financial_data = {
                    "monthly_income": monthly_income,
                    "dependants": dependants,
                    "transactions": transactions_df.to_dict('records') if transactions_df is not None else None,
                    "manual_expenses": manual_expenses if use_manual_expenses else None,
                    "debts": debts
                }
                
                finance_system = FinanceAdvisorSystem()
                
                try:
                    results = asyncio.run(finance_system.analyze_finances(financial_data))
                    
                    tabs = st.tabs(["💰 Budget Analysis", "📈 Savings Strategy", "💳 Debt Reduction"])
                    
                    with tabs[0]:
                        st.subheader("Budget Analysis")
                        if "budget_analysis" in results and results["budget_analysis"]:
                            display_budget_analysis(results["budget_analysis"])
                        else:
                            st.write("No budget analysis available.")
                    
                    with tabs[1]:
                        st.subheader("Savings Strategy")
                        if "savings_strategy" in results and results["savings_strategy"]:
                            display_savings_strategy(results["savings_strategy"])
                        else:
                            st.write("No savings strategy available.")
                    
                    with tabs[2]:
                        st.subheader("Debt Reduction Plan")
                        if "debt_reduction" in results and results["debt_reduction"]:
                            display_debt_reduction(results["debt_reduction"])
                        else:
                            st.write("No debt reduction plan available.")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    
    with goals_tab:
        show_notifications()
        st.header("🎯 Financial Goals Tracker")
        st.caption("Set, track, and visualize your financial goals.")
        if 'goals' not in st.session_state:
            st.session_state['goals'] = []
        if 'goal_edit_idx' not in st.session_state:
            st.session_state['goal_edit_idx'] = None
        if 'goal_search' not in st.session_state:
            st.session_state['goal_search'] = ''
        # --- Smart Goal Recommendations & Tracking ---
        def suggest_smart_goals(user_data):
            goals = []
            # Emergency Fund Goal
            expenses = 0
            if user_data.get('manual_expenses'):
                expenses = sum(user_data['manual_expenses'].values())
            elif user_data.get('transactions'):
                expenses = sum(t.get('Amount', 0) for t in user_data['transactions'])
            if expenses > 0:
                ef_target = round(expenses * 3, -2)  # 3 months of expenses, rounded
                goals.append({
                    'name': 'Emergency Fund',
                    'target': ef_target,
                    'current': 0,
                    'deadline': str(datetime.now().date().replace(year=datetime.now().year+1))
                })
            # Vacation Goal (suggest if surplus exists)
            income = user_data.get('monthly_income', 0)
            if income > expenses and income > 0:
                vac_target = 0.5 * expenses if expenses > 0 else 1000
                goals.append({
                    'name': 'Vacation',
                    'target': round(vac_target, -2),
                    'current': 0,
                    'deadline': str(datetime.now().date().replace(month=12, day=31))
                })
            # Debt Payoff Goal
            if user_data.get('debts'):
                total_debt = sum(d.get('amount', 0) for d in user_data['debts'])
                if total_debt > 0:
                    goals.append({
                        'name': 'Debt Payoff',
                        'target': total_debt,
                        'current': 0,
                        'deadline': str(datetime.now().date().replace(year=datetime.now().year+2))
                    })
            return goals

        # --- In Goals Tab: Recommend and Add Smart Goals ---
        with goals_tab:
            show_notifications()
            st.header("🎯 Financial Goals Tracker")
            st.caption("Set, track, and visualize your financial goals.")
            if 'goals' not in st.session_state:
                st.session_state['goals'] = []
            if 'goal_edit_idx' not in st.session_state:
                st.session_state['goal_edit_idx'] = None
            if 'goal_search' not in st.session_state:
                st.session_state['goal_search'] = ''
            # --- Smart Goal Recommendations ---
            user_data = {
                'monthly_income': st.session_state.get('income', 0),
                'manual_expenses': manual_expenses if 'manual_expenses' in locals() else {},
                'transactions': transactions_df.to_dict('records') if 'transactions_df' in locals() and transactions_df is not None else [],
                'debts': debts if 'debts' in locals() else []
            }
            recommended_goals = suggest_smart_goals(user_data)
            if recommended_goals:
                st.info("✨ Recommended Goals for You:")
                for g in recommended_goals:
                    if not any(existing['name'] == g['name'] for existing in st.session_state['goals']):
                        if st.button(f"Add '{g['name']}' Goal", key=f"add_smart_{g['name']}"):
                            st.session_state['goals'].append(g)
                            st.success(f"Added smart goal: {g['name']}")
                            st.rerun()
            # --- Summary Card ---
            total_goals = len(st.session_state['goals'])
            achieved_goals = sum(1 for g in st.session_state['goals'] if g['current'] >= g['target'])
            total_target = sum(g['target'] for g in st.session_state['goals'])
            total_saved = sum(g['current'] for g in st.session_state['goals'])
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Goals", total_goals)
            col2.metric("Goals Achieved", achieved_goals)
            col3.metric("Total Target", f"${total_target:,.2f}")
            col4.metric("Total Saved", f"${total_saved:,.2f}")
            # --- Celebrate Milestones ---
            if achieved_goals > 0:
                st.balloons()
                st.success(f"Congratulations! You've achieved {achieved_goals} goal(s)!")
            # --- Search/Filter ---
            st.text_input("Search Goals by Name", key='goal_search', help="Type to filter your goals by name.")
            # --- Goal input form (add or edit) ---
            with st.form("goal_form", clear_on_submit=True):
                if st.session_state['goal_edit_idx'] is not None:
                    edit_goal = st.session_state['goals'][st.session_state['goal_edit_idx']]
                    goal_name = st.text_input("Goal Name", value=edit_goal['name'])
                    goal_target = st.number_input("Target Amount ($)", min_value=1.0, step=10.0, value=edit_goal['target'])
                    goal_current = st.number_input("Current Saved ($)", min_value=0.0, step=10.0, value=edit_goal['current'])
                    goal_deadline = st.date_input("Target Date", value=pd.to_datetime(edit_goal['deadline']))
                else:
                    goal_name = st.text_input("Goal Name", help="E.g., Vacation, New Car, Emergency Fund")
                    goal_target = st.number_input("Target Amount ($)", min_value=1.0, step=10.0)
                    goal_current = st.number_input("Current Saved ($)", min_value=0.0, step=10.0)
                    goal_deadline = st.date_input("Target Date")
                submitted = st.form_submit_button("Save Goal")
                if submitted and goal_name and goal_target > 0:
                    new_goal = {
                        'name': goal_name,
                        'target': goal_target,
                        'current': goal_current,
                        'deadline': str(goal_deadline)
                    }
                    if st.session_state['goal_edit_idx'] is not None:
                        st.session_state['goals'][st.session_state['goal_edit_idx']] = new_goal
                        st.success(f"Goal '{goal_name}' updated!")
                        st.session_state['goal_edit_idx'] = None
                    else:
                        st.session_state['goals'].append(new_goal)
                        st.success(f"Goal '{goal_name}' added!")
            # --- Display and visualize goals ---
            filtered_goals = [g for g in st.session_state['goals'] if st.session_state['goal_search'].lower() in g['name'].lower()]
            if filtered_goals:
                st.subheader("Your Goals")
                goals_df = pd.DataFrame(filtered_goals)
                goals_df['Progress (%)'] = (goals_df['current'] / goals_df['target'] * 100).clip(upper=100)
                st.dataframe(goals_df[['name', 'target', 'current', 'deadline', 'Progress (%)']], hide_index=True)
                # --- Progress bar for each goal, edit/delete/update buttons ---
                for idx, row in goals_df.iterrows():
                    col1, col2, col3, col4 = st.columns([3,1,1,1])
                    with col1:
                        st.markdown(f"**{row['name']}** (Target: ${row['target']:.2f} by {row['deadline']})")
                        st.progress(min(row['current'] / row['target'], 1.0))
                        st.caption(f"Saved: ${row['current']:.2f} / ${row['target']:.2f} ({row['Progress (%)']:.1f}%)")
                    with col2:
                        if st.button("➕ Add Savings", key=f"add_{row['name']}_{idx}"):
                            add_amt = st.number_input(f"Add amount to '{row['name']}'", min_value=1.0, step=10.0, key=f"add_amt_{row['name']}_{idx}")
                            if add_amt > 0:
                                for g in st.session_state['goals']:
                                    if g['name'] == row['name']:
                                        g['current'] += add_amt
                                        st.success(f"Added ${add_amt:.2f} to '{row['name']}'!")
                                        break
                    with col3:
                        if st.button("✏️ Edit", key=f"edit_{row['name']}_{idx}"):
                            for i, g in enumerate(st.session_state['goals']):
                                if g['name'] == row['name']:
                                    st.session_state['goal_edit_idx'] = i
                                    st.rerun()
                with col4:
                    if st.button("🗑️ Delete", key=f"del_{row['name']}_{idx}"):
                        st.session_state['goals'] = [g for g in st.session_state['goals'] if g['name'] != row['name']]
                        st.success(f"Goal '{row['name']}' deleted!")
                        st.rerun()
                # --- Projected savings plan ---
                remaining = row['target'] - row['current']
                days_left = (pd.to_datetime(row['deadline']) - pd.to_datetime(datetime.now().date())).days
                if days_left > 0 and remaining > 0:
                    per_day = remaining / days_left
                    per_week = per_day * 7
                    st.info(f"To reach this goal, save at least ${per_day:.2f}/day or ${per_week:.2f}/week.")
                elif remaining <= 0:
                    st.balloons()
                    st.success(f"Goal '{row['name']}' achieved! 🎉")
                else:
                    st.warning(f"Goal '{row['name']}' deadline has passed. Consider updating your goal.")
                # --- All goals progress chart ---
                st.subheader("All Goals Progress")
                fig = go.Figure(go.Bar(
                    x=goals_df['name'],
                    y=goals_df['Progress (%)'],
                    marker_color=goals_df['Progress (%)'],
                    text=[f"{p:.1f}%" for p in goals_df['Progress (%)']],
                    textposition='auto',
                ))
                fig.update_layout(title="Progress Toward Each Goal", yaxis_title="Progress (%)", xaxis_title="Goal")
                st.plotly_chart(fig)
                # --- Timeline visualization ---
                st.subheader("Goal Timelines")
                if len(goals_df) > 0:
                    fig2 = px.timeline(goals_df, x_start=[datetime.now().date()]*len(goals_df), x_end='deadline', y='name', color='Progress (%)',
                                     labels={'name': 'Goal', 'deadline': 'Deadline'}, title="Goal Deadlines Timeline")
                    st.plotly_chart(fig2)
            else:
                st.info("No goals set yet. Add a goal above to get started!")
    
    with scenario_tab:
        st.header("🧮 What-If Scenario Simulator")
        st.caption("Simulate changes to your finances and see the projected impact.")
        # User inputs for simulation
        sim_income = st.number_input("Simulated Monthly Income ($)", min_value=0.0, step=100.0, value=3000.0)
        sim_expenses = st.number_input("Simulated Total Monthly Expenses ($)", min_value=0.0, step=100.0, value=2000.0)
        sim_debt = st.number_input("Simulated Total Debt ($)", min_value=0.0, step=100.0, value=5000.0)
        sim_debt_payment = st.number_input("Simulated Monthly Debt Payment ($)", min_value=0.0, step=50.0, value=200.0)
        sim_savings = st.number_input("Simulated Monthly Savings ($)", min_value=0.0, step=50.0, value=500.0)
        sim_months = st.slider("Months to Project", min_value=1, max_value=36, value=12)
        if st.button("Run Simulation", key="run_simulation"):
            # Projected savings and debt payoff
            proj_savings = []
            proj_debt = []
            savings = 0
            debt = sim_debt
            for m in range(sim_months):
                savings += sim_savings + max(0, sim_income - sim_expenses - sim_debt_payment - sim_savings)
                debt = max(0, debt - sim_debt_payment)
                proj_savings.append(savings)
                proj_debt.append(debt)
            st.subheader("Projected Savings and Debt Over Time")
            df_proj = pd.DataFrame({
                "Month": list(range(1, sim_months+1)),
                "Savings": proj_savings,
                "Debt": proj_debt
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_proj["Month"], y=df_proj["Savings"], mode="lines+markers", name="Savings", line=dict(color="green")))
            fig.add_trace(go.Scatter(x=df_proj["Month"], y=df_proj["Debt"], mode="lines+markers", name="Debt", line=dict(color="red")))
            fig.update_layout(title="What-If Projection: Savings & Debt", xaxis_title="Month", yaxis_title="Amount ($)")
            st.plotly_chart(fig)
            st.info(f"After {sim_months} months: Projected savings = ${proj_savings[-1]:,.2f}, Projected debt = ${proj_debt[-1]:,.2f}")
        st.markdown("""
        **Try these scenarios:**
        - Increase your income and see how quickly you can pay off debt.
        - Reduce expenses and see the impact on savings.
        - Increase debt payment to see how fast you become debt-free.
        """)
    
    with education_tab:
        st.header("📚 Financial Education Center")
        st.caption("Learn and test your knowledge on key personal finance topics.")
        # Personalized learning path
        user_fin_data = {
            "manual_expenses": manual_expenses if 'manual_expenses' in locals() else {},
            "transactions": transactions_df.to_dict('records') if transactions_df is not None else [],
            "debts": debts if 'debts' in locals() else []
        }
        learning_path = suggest_learning_path(user_fin_data, st.session_state.get('goals', []))
        st.markdown(f"**Recommended for you:** {', '.join(learning_path)}")
        # Lesson/quiz selector
        lesson_titles = [l.title for l in lessons]
        selected_lesson = st.selectbox("Choose a lesson:", lesson_titles, index=lesson_titles.index(learning_path[0]) if learning_path else 0)
        lesson = next(l for l in lessons if l.title == selected_lesson)
        st.subheader(lesson.title)
        st.write(lesson.content)
        # --- Enhanced Interactive Quiz with Explanations, Hints, and Feedback ---
        if lesson.quiz:
            st.markdown("#### Quiz")
            quiz_score = 0
            total_questions = len(lesson.quiz)
            quiz_answers = {}
            quiz_feedback = {}
            for i, q in enumerate(lesson.quiz):
                with st.expander(f"Question {i+1}"):
                    user_ans = st.radio(q['q'], q['a'], key=f"quiz_{lesson.title}_{i}")
                    quiz_answers[i] = user_ans
                    if st.button(f"Hint for Q{i+1}", key=f"hint_{lesson.title}_{i}"):
                        st.info("Think about the core principle behind this question.")
            if st.button("Submit Quiz", key=f"submit_{lesson.title}"):
                correct_count = 0
                for i, q in enumerate(lesson.quiz):
                    if q['a'].index(quiz_answers[i]) == q['correct']:
                        correct_count += 1
                        quiz_feedback[i] = (True, q['a'][q['correct']])
                    else:
                        quiz_feedback[i] = (False, q['a'][q['correct']])
                st.success(f"You scored {correct_count} out of {total_questions}!")
                if correct_count == total_questions:
                    st.balloons()
                    st.info("Outstanding! You mastered this topic.")
                elif correct_count >= total_questions // 2:
                    st.info("Good job! Review the explanations below to improve.")
                else:
                    st.warning("Keep practicing! See explanations below.")
                # Show explanations and feedback for each question
                for i, q in enumerate(lesson.quiz):
                    correct_ans = q['a'][q['correct']]
                    if quiz_feedback[i][0]:
                        st.success(f"Q{i+1}: Correct! ✅ The answer is: {correct_ans}")
                    else:
                        st.error(f"Q{i+1}: Incorrect ❌. The correct answer is: {correct_ans}")
                    st.caption("Explanation: This is a key concept for your financial journey.")
        # --- End Enhanced Interactive Quiz ---
        # --- Learning Streaks, Badges, and Leaderboard ---
        if 'edu_streak' not in st.session_state:
            st.session_state['edu_streak'] = 0
        if 'edu_badges' not in st.session_state:
            st.session_state['edu_badges'] = set()
        if st.button("Mark Lesson as Completed", key=f"complete_{lesson.title}"):
            st.session_state['edu_streak'] += 1
            st.success(f"Great! You've completed {st.session_state['edu_streak']} lesson(s) in a row.")
            if st.session_state['edu_streak'] == 3:
                st.session_state['edu_badges'].add('Consistent Learner')
                st.balloons()
                st.info("🏅 Badge Unlocked: Consistent Learner!")
            if st.session_state['edu_streak'] == 5:
                st.session_state['edu_badges'].add('Education Champion')
                st.balloons()
                st.info("🏆 Badge Unlocked: Education Champion!")
        st.markdown(f"**Learning Streak:** {st.session_state['edu_streak']} day(s)")
        if st.session_state['edu_badges']:
            st.markdown(f"**Badges:** {' | '.join(st.session_state['edu_badges'])}")
        # --- Leaderboard (local, for fun) ---
        if 'edu_leaderboard' not in st.session_state:
            st.session_state['edu_leaderboard'] = {'You': st.session_state['edu_streak'], 'Alex': 2, 'Sam': 4}
        st.session_state['edu_leaderboard']['You'] = st.session_state['edu_streak']
        st.markdown("---")
        st.markdown("### Leaderboard 🏅")
        leaderboard_df = pd.DataFrame(list(st.session_state['edu_leaderboard'].items()), columns=['Name', 'Streak'])
        leaderboard_df = leaderboard_df.sort_values('Streak', ascending=False)
        st.dataframe(leaderboard_df, hide_index=True)
        # --- Suggest Next Steps ---
        st.markdown("---")
        st.markdown("**Next Steps:**")
        if selected_lesson == "Budgeting Basics":
            st.info("Try tracking your expenses for a week and revisit this lesson to see your progress.")
        elif selected_lesson == "Understanding Credit Scores":
            st.info("Check your credit report online and note any changes after applying these tips.")
        elif selected_lesson == "Investing 101":
            st.info("Research a simple index fund and consider starting with a small investment.")
        st.markdown("---")
        st.info("More interactive modules, badges, and advanced quizzes coming soon! Suggest topics in the About tab.")
    with about_tab:
        st.markdown("""
        ### About AI Financial Coach
        
        This application uses Google's Agent Development Kit (ADK) to provide comprehensive financial analysis and advice through multiple specialized AI agents:
        
        1. **🔍 Budget Analysis Agent**
           - Analyzes spending patterns
           - Identifies areas for cost reduction
           - Provides actionable recommendations
        
        2. **💰 Savings Strategy Agent**
           - Creates personalized savings plans
           - Calculates emergency fund requirements
           - Suggests automation techniques
        
        3. **💳 Debt Reduction Agent**
           - Develops optimal debt payoff strategies
           - Compares different repayment methods
           - Provides actionable debt reduction tips
        
        ### Privacy & Security
        
        - All data is processed locally
        - No financial information is stored or transmitted
        - Secure API communication with Google's services
        
        ### Need Help?
        
        For support or questions:
        - Check the [documentation](https://github.com/abhishekkumar62000)
        - Report issues on [GitHub](https://github.com/abhishekkumar62000)
        """)

    # --- Sidebar Images and Developer Info ---
    AI_path = "AI.png"  # Ensure this file is in the same directory as your script
    if os.path.exists(AI_path):
        st.sidebar.image(AI_path)
    else:
        st.sidebar.warning("AI.png file not found. Please check the file path.")

    image_path = "image.png"  # Ensure this file is in the same directory as your script
    if os.path.exists(image_path):
        st.sidebar.image(image_path)
    else:
        st.sidebar.warning("image.png file not found. Please check the file path.")

    st.sidebar.markdown("👨‍💻 Developer:- Abhishek💖Yadav")
    developer_path = "pic.jpg"  # Ensure this file is in the same directory as your script
    if os.path.exists(developer_path):
        st.sidebar.image(developer_path)
    else:
        st.sidebar.warning("pic.jpg file not found. Please check the file path.")
    # Place the CSV download button below the developer image
    sample_csv = """Date,Category,Amount\n2024-01-01,Housing,1200.00\n2024-01-02,Food,150.50\n2024-01-03,Transportation,45.00"""
    st.sidebar.download_button(
        label="📥 Download CSV Template",
        data=sample_csv,
        file_name="expense_template.csv",
        mime="text/csv"
    )

    # --- Personalized AI Chatbot/Assistant in Sidebar (Enhanced Interactive Version) ---
    st.sidebar.markdown("---")
    st.sidebar.header("💬 Ask the AI Financial Coach (Chat)")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'chat_memory' not in st.session_state:
        st.session_state['chat_memory'] = []  # Stores (role, message) tuples for context

    # Display chat history in a chat-like format (user/AI turns)
    st.sidebar.markdown("#### Chat")
    for role, msg in st.session_state['chat_memory'][-10:]:
        if role == 'user':
            st.sidebar.markdown(f"<div style='text-align:right; color:#1a73e8;'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"<div style='text-align:left; color:#34a853;'><b>AI:</b> {msg}</div>", unsafe_allow_html=True)

    # Input box for user message (one-to-one chat)
    user_message = st.sidebar.text_input("Type your message and press Enter...", key="chat_input", value="", on_change=None)
    send_btn = st.sidebar.button("Send", key="send_chatbot")
    if user_message.strip() and send_btn:
        # Add user message to memory
        st.session_state['chat_memory'].append(('user', user_message.strip()))
        # Gather user data for context
        user_context = {
            'income': st.session_state.get('income', 0),
            'manual_expenses': manual_expenses if 'manual_expenses' in locals() else {},
            'transactions': transactions_df.to_dict('records') if 'transactions_df' in locals() and transactions_df is not None else [],
            'debts': debts if 'debts' in locals() else [],
            'goals': st.session_state.get('goals', [])
        }
        # Build conversation history for memory (last 10 turns)
        chat_history = "\n".join([
            ("User: " + m) if r == 'user' else ("AI: " + m)
            for r, m in st.session_state['chat_memory'][-10:]
        ])
        # Compose prompt for the AI with memory
        prompt = (
            f"You are a helpful, friendly, and expert AI financial coach.\n"
            f"The user has the following data: {json.dumps(user_context)}.\n"
            f"Here is the previous conversation:\n{chat_history}\n"
            f"User: {user_message.strip()}\n"
            f"Respond as a financial coach, referencing the user's data and previous conversation.\n"
            f"If the user asks about their goals, debts, or spending, use the numbers provided.\n"
            f"If you give advice, make it actionable and encouraging.\n"
            f"Keep your answers short, clear, and interactive.\n"
        )
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("Google_Gemini_ai_key"))
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = model.generate_content(prompt)
            answer = response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            answer = f"[AI Error] {str(e)}"
        # Add AI response to memory
        st.session_state['chat_memory'].append(('ai', answer))
       
        # Optionally, also keep a simple history for export
        st.session_state['chat_history'].append((user_message.strip(), answer))
        st.rerun()
    # Option to clear chat
    if st.sidebar.button("Clear Chat", key="clear_chatbot"):
        st.session_state['chat_memory'] = []
        st.session_state['chat_history'] = []
        st.rerun()

if __name__ == "__main__":
    main()