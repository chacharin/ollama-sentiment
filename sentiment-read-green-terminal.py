# นำเข้าไลบรารีที่ใช้ในโปรแกรม
import ollama  # ไลบรารีสำหรับติดต่อกับโมเดล AI ของ Ollama
from langchain.prompts import PromptTemplate  # เครื่องมือสร้างโครงสร้างของคำสั่ง (Prompt)

# ตั้งค่า Prompt เพื่อกำหนดแนวทางการตอบของแชทบอท
prompt_template = PromptTemplate.from_template(
    """
    [System Instructions]
    You are an AI assistant specialized in evaluating sentiment. 
    Your job is to determine whether a given sentence is positive or negative.

    [Sentiment Analysis Guidelines]
    - If the sentence is positive, respond with "green".
    - If the sentence is negative, respond with "red".
    - If the sentiment is unclear, respond with "neutral".

    [User Query]
    - Sentence: {input}

    [Response Format]
    - Only respond with "green", "red", or "neutral".

    [Example Responses]
    **Scenario 1: Positive Sentence**
    - Input: "I love this robot!"
    - Output: green

    **Scenario 2: Negative Sentence**
    - Input: "This robot is terrible."
    - Output: red

    **Scenario 3: Neutral Sentence**
    - Input: "The robot is on the table."
    - Output: neutral
    """
)

# ฟังก์ชันสำหรับรับข้อความตอบกลับจาก AI ทีละส่วน (streaming response)
def model_response_generator(input_text):
    # ปรับแต่งข้อความตาม Prompt Template
    formatted_prompt = prompt_template.format(input=input_text)
    
    # สนทนากับโมเดลโดยส่งข้อความที่มีอยู่ไปให้ AI
    stream = ollama.chat(
        model="llama3.1:8b",  # ใช้โมเดลจาก Ollama
        messages=[{"role": "user", "content": formatted_prompt}],  # ประวัติข้อความในการสนทนา
        stream=True,  # เปิดใช้งานการสตรีมข้อมูลทีละส่วน
    )
    full_response = ""  # ตัวแปรสำหรับเก็บข้อความที่ได้รับทั้งหมด
    for chunk in stream:
        full_response += chunk["message"]["content"]  # สะสมข้อความที่ได้รับทีละส่วน
    return full_response  # ส่งคืนข้อความที่ได้รับทั้งหมด

# ประวัติการสนทนา
messages = []

# ต้อนรับผู้ใช้
print("ยินดีต้อนรับสู่ระบบสนทนา AI!")
print("คุณสามารถถามคำถามหรือพิมพ์ 'exit' เพื่อออกจากโปรแกรม.")

# วนลูปเพื่อสนทนากับผู้ใช้
while True:
    # รับข้อความจากผู้ใช้
    user_input = input("\nคุณ: ")
    
    # ตรวจสอบคำสั่งออกจากโปรแกรม
    if user_input.lower() == "exit":
        print("ขอบคุณที่ใช้บริการ!")
        break

    # เพิ่มข้อความผู้ใช้ลงในประวัติการสนทนา
    messages.append({"role": "user", "content": user_input})

    # รับคำตอบจาก AI
    print("\nAI: กำลังคิด...")
    full_response = model_response_generator(user_input)
    print(f"AI: {full_response}")

    # เพิ่มคำตอบจาก AI ลงในประวัติการสนทนา
    messages.append({"role": "assistant", "content": full_response})