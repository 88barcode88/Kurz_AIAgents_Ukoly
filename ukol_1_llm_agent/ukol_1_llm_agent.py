import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Jednoduchý nástroj: faktoriál
def factorial(n: int) -> dict:
    result = 1
    for i in range(2, n+1):
        result *= i
    return {"input": n, "factorial": result}

# Nástrojová definice pro OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "factorial",
            "description": "Calculate factorial of a number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number to calculate factorial for"}
                },
                "required": ["n"],
            }
        }
    }
]

available_functions = {
    "factorial": factorial,
}

class ReactAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.max_iterations = 5
    
    def run(self, messages: List[Dict[str, Any]]) -> str:
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False
            )
            
            response_message = response.choices[0].message
            print(f"LLM Response: {response_message}")
            
            if response_message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in response_message.tool_calls
                    ]
                })
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id
                    
                    print(f"Calling tool: {function_name} with args {function_args}")
                    
                    func = available_functions[function_name]
                    func_response = func(**function_args)
                    
                    print(f"Tool response: {func_response}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": function_name,
                        "content": json.dumps(func_response),
                    })
                
                continue
            
            else:
                final_content = response_message.content
                messages.append({
                    "role": "assistant",
                    "content": final_content,
                })
                print(f"\nFinal answer: {final_content}")
                return final_content
        
        return "Error: Max iterations reached."

def main():
    agent = ReactAgent()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can calculate factorials."},
        {"role": "user", "content": "Calculate the factorial of 6."},
    ]
    
    result = agent.run(messages)
    print(f"\nResult:\n{result}")

if __name__ == "__main__":
    main()
