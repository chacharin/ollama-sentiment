# นำเข้าไลบรารีที่ใช้ในโปรแกรม
import streamlit as st  # ไลบรารีสำหรับสร้างเว็บแอปพลิเคชันแบบโต้ตอบ
import ollama  # ไลบรารีสำหรับติดต่อกับโมเดล AI ของ Ollama
from langchain.prompts import PromptTemplate  # เครื่องมือสร้างโครงสร้างของคำสั่ง (Prompt)

# ตั้งค่าหน้าเว็บของ Streamlit เช่น ชื่อและการจัดวางองค์ประกอบ
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("ระบบสนทนากับ AI")  # แสดงหัวข้อของหน้าเว็บ

# ตรวจสอบว่ามีประวัติข้อความใน session state หรือไม่ ถ้าไม่มีให้สร้างค่าว่าง
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # เก็บข้อความสนทนาไว้ใน session state

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
    # ส่งคืนข้อความทีละส่วนที่ได้จากการสตรีม
    for chunk in stream:
        yield chunk["message"]["content"]

# แสดงประวัติการสนทนาบนหน้าจอ 
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):  # แสดงข้อความตามบทบาท เช่น ผู้ใช้หรือผู้ช่วย AI
        st.markdown(message["content"])  # แสดงข้อความในรูปแบบ Markdown

# รับข้อความใหม่จากผู้ใช้ผ่านช่องป้อนข้อมูล
user_input = st.chat_input("ป้อนข้อความที่คุณต้องการถาม AI:")  # ให้ผู้ใช้ป้อนคำถามหรือข้อความ
if user_input:  # ตรวจสอบว่าผู้ใช้ได้ป้อนข้อความหรือไม่
    # เพิ่มข้อความที่ผู้ใช้ป้อนเข้ามาในประวัติข้อความ
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # แสดงข้อความที่ผู้ใช้ป้อนบนหน้าจอ
    with st.chat_message("user"):
        st.markdown(user_input)

    # เตรียมพร้อมการแสดงผลลัพธ์จาก AI
    with st.chat_message("assistant"):
        status_placeholder = st.empty()  # ช่องว่างสำหรับแสดงสถานะการทำงานของ AI
        response_placeholder = st.empty()  # ช่องว่างสำหรับแสดงข้อความตอบกลับ
        status_placeholder.markdown("🤔 **AI กำลังคิด...**")  # แสดงสถานะขณะรอคำตอบจาก AI

        # สตรีมข้อความตอบกลับจาก AI และแสดงข้อความบนหน้าจอ
        streamed_response = ""  # ตัวแปรสำหรับเก็บข้อความที่ได้รับ
        for chunk in model_response_generator(user_input):
            status_placeholder.empty()  # ลบข้อความสถานะเมื่อ AI เริ่มตอบกลับ
            streamed_response += chunk  # รวมข้อความที่ได้รับทีละส่วนเข้าด้วยกัน
            response_placeholder.markdown(streamed_response)  # แสดงข้อความที่ได้รับทั้งหมดจนถึงปัจจุบัน
        
        print(str(streamed_response))

        # บันทึกข้อความตอบกลับจาก AI ลงในประวัติข้อความ
        st.session_state["messages"].append({"role": "assistant", "content": streamed_response})