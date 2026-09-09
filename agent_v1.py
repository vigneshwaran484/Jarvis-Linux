import requests
import subprocess
import webbrowser
from config import ALLOWED_APPS
# ─────────────────────────────────────────
# 1. THE BRAIN
# ─────────────────────────────────────────

def ask_llm(messages):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": messages,
            "stream": False
        }
    )
    return response.json()["message"]["content"]


# ─────────────────────────────────────────
# 2. THE TOOLS
# ─────────────────────────────────────────

def open_app(app_name):
    app = ALLOWED_APPS.get(app_name.lower())
    if app:
        subprocess.Popen(app)
        return f"Opened {app_name} successfully"
    return f"I don't know how to open {app_name}"
def close_app(app_name):
    app_name = app_name.strip().lower()
    process = ALLOWED_APPS.get(app_name)
    if not process:
        # Derive from name if not in map
        process = app_name + ".exe"
    # Use /im (image name) only — no shell injection risk since we control the string
    result = subprocess.run(
        ["taskkill", "/f", "/im", process],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return f"Closed {app_name}"
    return f"Could not close {app_name} — it may not be running."

def search_web(query):
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return f"Opened Google search for: {query}"

def write_file(filename, content):
    with open(filename, "w") as f:
        f.write(content)
    return f"File '{filename}' created successfully"

def read_file(filename):
    try:
        with open(filename, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File '{filename}' not found"

# Map names to actual functions
TOOLS = {
    "open_app":   open_app,
    "close_app":  close_app,
    "search_web": search_web,
    "write_file": write_file,
    "read_file":  read_file,
}


# ─────────────────────────────────────────
# 3. THE SYSTEM PROMPT
# ─────────────────────────────────────────

SYSTEM_PROMPT = """You are Jarvis, a personal AI assistant running locally.

You have access to these tools:
- open_app(app_name)         → opens notepad, calculator, chrome, explorer
- search_web(query)          → searches Google in the browser
- write_file(filename, content) → creates a text file
- read_file(filename)        → reads a text file

RULES:
- When you need to use a tool, respond EXACTLY in this format:
  TOOL: tool_name|argument1|argument2

- When you are completely done, respond EXACTLY like this:
  DONE: your final message to the user

- Only call ONE tool per response.
- After a tool result is given, decide if you need another tool or if you're done.

Example:
User: open notepad
You: TOOL: open_app|notepad

User: Tool result: Opened notepad successfully
You: DONE: Notepad is now open for you!
"""


# ─────────────────────────────────────────
# 4. THE AGENT LOOP
# ─────────────────────────────────────────

def run_agent(user_goal):
    print(f"\n🎯 Goal: {user_goal}")
    print("─" * 40)

    # Start conversation with system prompt + user goal
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_goal}
    ]

    # Safety limit — max 10 steps so it doesn't loop forever
    for step in range(10):

        # Ask LLM what to do
        response = ask_llm(messages)
        print(f"\n🧠 Jarvis: {response}")

        # Add LLM response to memory
        messages.append({"role": "assistant", "content": response})

        # ── Did it want to use a tool? ──
        if response.strip().startswith("TOOL:"):
            # Parse: "TOOL: open_app|notepad"
            parts = response.strip().replace("TOOL:", "").strip().split("|")
            tool_name = parts[0].strip()
            tool_args = parts[1:]  # rest are arguments

            if tool_name in TOOLS:
                print(f"🔧 Running tool: {tool_name} with args {tool_args}")

                # Call the tool with the right number of arguments
                result = TOOLS[tool_name](*tool_args)
                print(f"✅ Tool result: {result}")

                # Feed result back into conversation
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {result}"
                })

            else:
                # Tool not found
                messages.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' does not exist. Try a different approach."
                })

        # ── Is it done? ──
        elif response.strip().startswith("DONE:"):
            final = response.strip().replace("DONE:", "").strip()
            print(f"\n✅ Jarvis: {final}")
            return final

        # ── Still thinking ──
        else:
            messages.append({
                "role": "user",
                "content": "Continue with the task."
            })

    print("⚠️  Reached max steps")
    return "Task incomplete"


# ─────────────────────────────────────────
# 5. MAIN — chat loop
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Jarvis is ready! (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Jarvis: Goodbye!")
            break

        run_agent(user_input)