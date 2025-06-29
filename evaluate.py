# evaluate.py
from rag_system import build_rag
import time

QUESTIONS = [

    {
        "q": "What qualifications and documents do Ugandan students need to apply for a CS bachelor's or master's in Germany?",
        "gt": "For Bachelor's Programs, Ugandan students need a recognized university entrance qualification and language proficiency proof (German for German-taught programs or English proficiency via IELTS/TOEFL for English-taught programs). Additional documentation may include a statement of purpose, CV, and passport photocopy, though requirements vary by institution.\nFor Master's Programs, students need all bachelor's requirements plus a bachelor's degree with sufficient merit in computer science, informatics, or related field, academic letters of recommendation from faculty, GRE General Test scores, and portfolio evidence such as computer science competition participation or completed MOOC certificates. Requirements vary significantly between institutions and specific programs, so always verify with your target universities."
    },

    {
        "q": "Will my Ugandan bachelor's degree be accepted for graduate studies in Germany?",
        "gt": "Recognition depends on the type of higher education institution and degree you're pursuing. Students with a completed 3-year bachelor's degree can directly apply for general admission to any subject area, provided they studied full-time at a recognized higher institution according to study regulations. Many universities work with uni-assist to check and approve foreign university entrance qualifications. However, some technical universities in Baden-Württemberg work with the Studienkolleg at University Konstanz instead of uni-assist. Check your specific university's requirements and procedures."
    },
    {
        "q": "Which exams can I take to prove German proficiency for CS programs in Germany?",
        "gt": "For German-taught computer science programs, you can meet language requirements through several options: TestDaF with minimum score of 4 in all parts, Goethe Certificate C1 or C2, German Language Diploma (DSD II) from the Standing Conference of Ministers of Education, passing the German section of a Studienkolleg assessment test, telc C1 Hochschule exam, DSH-2 or above, ÖSD C2, German Language Test II from Munich Language and Interpreting Institute, or a degree (Bachelor/Master/Diplom) where courses were taught in German."
    },
]

# Build RAG system
qa = build_rag()

results = []

# Run questions
for item in QUESTIONS:
    start = time.time()
    response = qa.invoke(item["q"])
    # Extract answer text from response dict
    answer = response.get("result", "") or response.get("text", "") or ""
    answer = answer.strip()
    duration = time.time() - start
    results.append({"question": item["q"], "answer": answer, "duration": duration})

# Display results
for result in results:
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Duration: {result['duration']:.2f} seconds")
    print("-" * 50)

 







