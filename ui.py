# ui.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AstraDoc UI", page_icon="📄", layout="wide")
st.title("AstraDoc AI — Upload document (auto-send to predefined emails)")

st.markdown("""
This UI only asks you to upload a file. The backend uses the predefined department email map
and will automatically send combined messages to those addresses (or skip if SMTP not configured).
""")

uploaded_file = st.file_uploader("Choose a file (pdf, docx, txt)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    if st.button("Process & Send (uses server predefined emails)"):
        with st.spinner("Processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                resp = requests.post(f"{API_URL}/classify", files=files, timeout=180)
                if resp.status_code != 200:
                    st.error(f"Backend error: {resp.status_code}")
                    st.text(resp.text)
                else:
                    payload = resp.json()
                    combined = payload.get("combined_messages", [])
                    emails_sent = payload.get("emails_sent", [])
                    st.success("Processing complete — combined messages generated.")
                    # Show combined messages
                    st.markdown("### 📬 Combined messages (department + email + message)")
                    for msg in combined:
                        dept = msg.get("department", "Unknown")
                        subject = msg.get("subject", "")
                        message = msg.get("message", msg.get("body", ""))
                        # get predefined email
                        # (Backend already used this map to send — but show here as info)
                        st.markdown(f"**Department:** {dept}")
                        st.markdown(f"**Subject:** {subject}")
                        st.text_area(f"msg_{dept}", value=message, height=200, key=f"msg_{dept}_view")
                        st.markdown("---")
                    # Show email sending results
                    st.markdown("### ✉️ Email send results")
                    for e in emails_sent:
                        dept = e.get("department")
                        email = e.get("email")
                        status = e.get("status")
                        error = e.get("error")
                        if status == "sent":
                            st.success(f"{dept} -> {email} : SENT")
                        elif status == "skipped":
                            st.warning(f"{dept} -> {email} : SKIPPED ({error})")
                        else:
                            st.error(f"{dept} -> {email} : ERROR ({error})")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request to backend failed: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

st.caption("AstraDoc — backend sends mails using server-side predefined DEPT_EMAIL_MAP")
