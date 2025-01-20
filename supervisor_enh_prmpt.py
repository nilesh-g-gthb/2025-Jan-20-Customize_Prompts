from utils_qnt_fx_warng_1 import LLMHandler
from utils_qnt_fx_warng_1 import MODEL_ID

# Enhanced Fixed Income Classification Instructions
CLASSIFICATION_INSTRUCTIONS = """Instruction: You are a supervisor tasked with managing chat messages about Fixed Income instruments.
Your task is to classify incoming questions into exactly three categories.
This classification is crucial for routing questions to the right team.

The three categories are:

1. QuoteRequest
   - Contains specific security details (ISIN, rate, maturity)
   - Asks for price, bid, or offer
   - Usually includes quantity (Qtm, size, amount)
   - Examples of phrases: "quote please", "offer please", "bid wanted", "price please"

2. BondRequest
   - Asks for information about bonds/securities
   - Research or analysis related queries
   - Questions about bond features, credit rating, etc.
   - No specific price/quote request

3. GENERAL
   - Any other general questions
   - Market commentary requests
   - Process or policy related questions

Return in the output only one word (QuoteRequest, BondRequest or GENERAL).

Examples:
Chat Message: 8.3774% HDB Financial Apr 26 INE756I07ER5 Qtm: 1 Cr Offer please
Output:QuoteRequest

Chat Message: SBI 7.72% 2025 bonds quote wanted size 5cr
Output:QuoteRequest

Chat Message: Need offer for HDFC Bank tier 2 bonds 500k size
Output:QuoteRequest

Chat Message: can you please share a brief note about AP state bonds?
Output:BondRequest

Chat Message: What's the credit rating of REC Limited bonds?
Output:BondRequest

Chat Message: How is the track record of profitability?
Output:BondRequest

Chat Message: What's your view on interest rates?
Output:GENERAL

Chat Message: How do I open a trading account?
Output:GENERAL

Chat Message: """

def create_prompt(user_input: str) -> str:
    """Create the complete prompt by combining the instruction and user input."""
    return CLASSIFICATION_INSTRUCTIONS + user_input

def main():
    """Main function to run the chatbot."""
    print(f"Fixed Income Query Classification Bot using {MODEL_ID} (type 'exit' to quit)")
    print("-" * 50)
    
    # Initialize LLM
    llm = LLMHandler()
    try:
        llm.initialize_llm()
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        return
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter your query: ").strip()
            
            # Check for exit command
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Create prompt and get response
            prompt = create_prompt(user_input)
            response = llm.get_llm_response(prompt)
            
            # Print response
            print(f"\nClassification: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nGENERAL")  # Fallback response for any unexpected errors
            continue

if __name__ == "__main__":
    main()