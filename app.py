import os
import gradio as gr
import requests
import json
import base64

N8N_WEBHOOK_URL = "https://nafay-ai.app.n8n.cloud/webhook/gpa-agent"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&display=swap');
:root {
    --bg-main: #020617; /* Very Deep Navy */
    --bg-card: #0f172a; 
    --border-color: #1e293b; 
    --text-main: #f8fafc; 
    --text-muted: #94a3b8;
    --accent: #06b6d4; /* Neon Cyan */
    --accent-hover: #0891b2;
}
body, .gradio-container { 
    background-color: var(--bg-main) !important; 
    font-family: 'Chakra Petch', sans-serif !important; 
    color: var(--text-main) !important; 
}
/* 1. Hero Section (Alignment Fixed) */
.hero {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    padding: 50px 20px;
    background: radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.15) 0%, transparent 70%);
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 24px;
}
.hero-tag {
    background: rgba(6, 182, 212, 0.1);
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.hero-title { 
    font-size: 38px !important; 
    font-weight: 700 !important; 
    margin: 0 0 10px !important; 
    text-shadow: 0 0 10px rgba(6, 182, 212, 0.3);
}
.hero-sub { 
    font-size: 15px; 
    color: var(--text-muted); 
    max-width: 600px; 
    margin: 0 auto !important; 
    text-align: center !important;
}
/* 2. Inputs & Forms */
label span { font-size: 12px !important; color: var(--accent) !important; text-transform: uppercase; }
textarea, input[type="text"], input[type="number"] { 
    background: #020617 !important; 
    border: 1px solid var(--border-color) !important; 
    border-radius: 8px !important; 
    padding: 12px !important;
    color: var(--text-main) !important;
}
textarea:focus, input:focus { 
    border-color: var(--accent) !important; 
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.2) !important; 
    outline: none !important;
}
/* 3. Primary Button */
#submit-btn {
    background: transparent !important;
    border: 2px solid var(--accent) !important; 
    border-radius: 8px !important;
    font-weight: 700 !important; 
    font-size: 16px !important; 
    padding: 14px !important; 
    color: var(--accent) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease !important;
}
#submit-btn:hover { 
    background: var(--accent) !important; 
    color: #000000 !important; 
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.5) !important;
}
.wrap, .panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}
.tip-box { background: rgba(6, 182, 212, 0.05); border-left: 3px solid var(--accent); padding: 12px; font-size: 13px; color: #cbd5e1; margin-bottom: 16px; }
.standing-good { color: #10b981 !important; }
.standing-warn { color: #f59e0b !important; }
.standing-bad  { color: #ef4444 !important; }
"""

# Grade to GPA points (letter grade input)
GRADE_POINTS = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F": 0.0
}

# Grade to representative marks (for n8n math engine compatibility)
GRADE_TO_MARK = {
    "A+": 92, "A": 87, "A-": 82,
    "B+": 77, "B": 72, "B-": 69,
    "C+": 65, "C": 61, "C-": 58,
    "D+": 54, "D": 51, "D-": 48,
    "F": 0
}


# ── Vision OCR ────────────────────────────────────────────────────────────────

def extract_from_image(image_path):
    """Send transcript image to Groq Vision, extract past CGPA + courses."""
    if not image_path:
        return None, "No image provided."
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY not set in Space secrets."

    try:
        mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "You are a transcript parser. Extract data from this university transcript image.\n"
                        "Return ONLY a raw JSON object — no markdown, no explanation, no code blocks.\n"
                        "Start your response with '{' and end with '}'.\n\n"
                        "Required JSON format:\n"
                        "{\n"
                        "  \"past_cgpa\": <float>,\n"
                        "  \"past_credits\": <float>,\n"
                        "  \"past_courses\": [\n"
                        "    {\"name\": \"<course name>\", \"credits\": <float>, \"grade\": \"<letter grade like A, B+, C>\"}\n"
                        "  ]\n"
                        "}\n\n"
                        "Rules:\n"
                        "- Use letter grades only (A, A-, B+, B, B-, C+, C, C-, D+, D, F)\n"
                        "- If CGPA not found, use 0\n"
                        "- If credits not found, use 0\n"
                        "- Include ALL courses visible in the transcript"
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                ]
            }],
            "temperature": 0.0,
            "max_tokens": 2000
        }

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=30
        )
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Clean any accidental markdown
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        start = content.find("{")
        end   = content.rfind("}") + 1
        if start == -1 or end == 0:
            return None, f"Vision model returned invalid JSON:\n{content[:300]}"

        parsed = json.loads(content[start:end])
        return parsed, None

    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except Exception as e:
        return None, f"Vision OCR error: {e}"


# ── Course text parser (robust) ───────────────────────────────────────────────

def parse_manual_courses(text):
    """
    Parse current semester courses from free-form text.
    Supports formats:
      Data Structures 3 C
      Machine Learning, 3, B+
      HCI 3 75       (marks format)
    Returns list of {name, credits, marks} for n8n.
    """
    courses = []
    if not text or not text.strip():
        return courses

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Normalize: replace commas/tabs with spaces
        line = line.replace(",", " ").replace("\t", " ")
        parts = line.split()
        if len(parts) < 3:
            continue

        # Last token = grade or marks, second-to-last = credits
        last   = parts[-1].upper().strip()
        second = parts[-2].strip()

        # Validate credits
        try:
            credits = float(second)
        except ValueError:
            continue

        # Resolve marks
        if last in GRADE_TO_MARK:
            marks = GRADE_TO_MARK[last]
        else:
            try:
                marks = float(last)
                if not (0 <= marks <= 100):
                    continue
            except ValueError:
                continue

        name = " ".join(parts[:-2]).strip()
        if not name:
            continue

        courses.append({"name": name, "credits": credits, "marks": marks})

    return courses


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_pipeline(student_name, past_cgpa_manual, past_credits_manual,
                     transcript_img, manual_courses_text):

    if not student_name.strip():
        return "Please enter your name.", "", "No data provided.", ""

    logs = []

    # 1. Extract past data
    past_courses        = []
    final_past_cgpa     = float(past_cgpa_manual or 0)
    final_past_credits  = float(past_credits_manual or 0)
    vision_log          = ""

    if transcript_img:
        vision_data, vision_err = extract_from_image(transcript_img)
        if vision_err:
            vision_log = f"Vision OCR warning: {vision_err}\nUsing manual values instead."
            logs.append(vision_log)
        else:
            final_past_cgpa    = float(vision_data.get("past_cgpa", past_cgpa_manual) or 0)
            final_past_credits = float(vision_data.get("past_credits", past_credits_manual) or 0)

            for c in vision_data.get("past_courses", []):
                grade = str(c.get("grade", "F")).upper().strip()
                marks = GRADE_TO_MARK.get(grade, 0)
                past_courses.append({
                    "name":    c.get("name", "Unknown"),
                    "credits": float(c.get("credits", 3)),
                    "marks":   marks
                })

            logs.append(f"Vision OCR: extracted {len(past_courses)} past courses, CGPA={final_past_cgpa}, Credits={final_past_credits}")

    # 2. Parse current semester
    current_courses = parse_manual_courses(manual_courses_text)
    if not current_courses:
        logs.append("No current semester courses found.")

    if not current_courses and not past_courses:
        return (
            "Please enter current semester courses in the text box.\n"
            "Format: Course Name, Credits, Grade\n"
            "Example: Data Structures, 3, C",
            "", "No data.", "\n".join(logs)
        )

    # 3. Send to n8n
    payload = {
        "name":           student_name.strip(),
        "pastCgpa":       final_past_cgpa,
        "pastCredits":    final_past_credits,
        "pastCourses":    past_courses,
        "currentCourses": current_courses
    }

    logs.append(f"Sending to n8n: {len(past_courses)} past + {len(current_courses)} current courses")

    try:
        resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return "Cannot connect to n8n workflow. Check if workflow is active.", "", "", "\n".join(logs)
    except requests.exceptions.Timeout:
        return "n8n request timed out (45s). Try again.", "", "", "\n".join(logs)
    except Exception as e:
        return f"Error: {e}", "", "", "\n".join(logs)

    # 4. Parse response
    sem_gpa          = data.get("semesterGpa", "N/A")
    cgpa             = data.get("cgpa", "N/A")
    standing         = data.get("standing", "N/A")
    courses_r        = data.get("courses", [])
    ai_advice        = data.get("aiAdvice", "No advice generated.")
    scenarios        = data.get("detailedScenarios", "No scenarios generated.")
    escalation       = data.get("escalation", False)
    failed_courses   = data.get("failedCourses", [])
    escalation_reason= data.get("escalationReason", "")

    # 5. Build main report
    report = f"""TRANSCRIPT STRATEGIC AUDIT
{"=" * 45}
Student:      {student_name}
Semester GPA: {sem_gpa} / 4.0
Overall CGPA: {cgpa} / 4.0
Standing:     {standing}
"""

    if escalation:
        report += f"\nESCALATION REQUIRED\n{'-' * 45}\n{escalation_reason}\n"

    if failed_courses:
        report += f"\nFAILED COURSES: {', '.join(failed_courses)}\n"

    report += f"\nCURRENT SEMESTER BREAKDOWN\n{'-' * 45}\n"
    for c in courses_r:
        report += f"  {c.get('name',''):<28} {c.get('credits','')} Cr   {c.get('grade','')}\n"

    if past_courses:
        report += f"\n(+ {len(past_courses)} past courses analyzed for what-if scenarios)\n"

    report += f"\nAI ADVISOR ANALYSIS\n{'-' * 45}\n{ai_advice}"

    logs.append(f"Response received: CGPA={cgpa}, Standing={standing}, Escalation={escalation}")

    return report, standing, scenarios, "\n".join(logs)


# ── UI ─────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(css=CSS, title="AI Transcript Strategist") as demo:

        gr.HTML("""
        <div class="hero">
            <div class="hero-tag">AI Academic Advisor</div>
            <h1 class="hero-title">AI Transcript <span>Strategist</span></h1>
            <p class="hero-sub">
                Upload your transcript. Enter current semester grades.
                Get your CGPA, academic standing, and a personalized strategy
                to improve — powered by n8n and Groq AI.
            </p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left: Inputs ──────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):

                student_name = gr.Textbox(
                    label="Student Name",
                    placeholder="e.g. Nafay"
                )

                gr.HTML("<div class='divider'></div>")
                gr.HTML("<div class='section-label'>Past Academic History</div>")
                gr.HTML("""
                <div class='tip-box'>
                    Upload your old transcript image. The AI will automatically
                    extract your Past CGPA and courses using Vision OCR.
                </div>
                """)

                transcript_img = gr.Image(
                    type="filepath",
                    label="Upload Past Transcript (optional)"
                )

                with gr.Row():
                    past_cgpa = gr.Number(
                        label="Manual Past CGPA",
                        value=0.0,
                        minimum=0.0,
                        maximum=4.0,
                        info="Used if image upload fails"
                    )
                    past_credits = gr.Number(
                        label="Manual Past Credits",
                        value=0,
                        minimum=0,
                        info="Total credits completed so far"
                    )

                gr.HTML("<div class='divider'></div>")
                gr.HTML("<div class='section-label'>Current Semester Courses</div>")
                gr.HTML("""
                <div class='tip-box'>
                    <b>Format:</b> Course Name, Credits, Grade (one per line)<br>
                    <b>Example:</b><br>
                    Data Structures, 3, C<br>
                    Machine Learning, 3, B+<br>
                    HCI, 3, A-
                </div>
                """)

                manual_courses = gr.Textbox(
                    label="Current Semester Courses",
                    placeholder="Data Structures, 3, C\nMachine Learning, 3, B+\nHCI, 3, A-",
                    lines=6
                )

                gr.HTML("<div class='divider'></div>")

                submit_btn = gr.Button(
                    "Run Strategic Audit →",
                    variant="primary",
                    elem_id="submit-btn"
                )

            # ── Right: Outputs ────────────────────────────────────────────────
            with gr.Column(scale=2):

                standing_output = gr.Textbox(
                    label="Academic Standing",
                    interactive=False,
                    max_lines=1
                )

                with gr.Accordion("What-If Scenarios: Improvement Roadmap", open=True):
                    scenarios_output = gr.Textbox(
                        label="",
                        lines=14,
                        interactive=False,
                        placeholder="Improvement scenarios will appear here..."
                    )

                result_output = gr.Textbox(
                    label="Full Audit Report",
                    lines=18,
                    interactive=False,
                    placeholder="Your complete audit report will appear here..."
                )

                with gr.Accordion("Debug Log", open=False):
                    log_output = gr.Textbox(
                        label="",
                        lines=6,
                        interactive=False,
                        placeholder="Pipeline logs will appear here..."
                    )

        submit_btn.click(
            fn=process_pipeline,
            inputs=[student_name, past_cgpa, past_credits, transcript_img, manual_courses],
            outputs=[result_output, standing_output, scenarios_output, log_output]
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)