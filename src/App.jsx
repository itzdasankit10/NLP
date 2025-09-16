import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquare, Loader, Download, BookOpen, RefreshCw } from "lucide-react";

const API_URL = "http://localhost:8000";

const subjectOptions = {
  '6': ['Geography', 'History', 'Economics', 'English', 'Biology', 'Political Science'],
  '7': ['Geography', 'History', 'Economics', 'English', 'Biology', 'Political Science'],
  '8': ['Geography', 'History', 'Economics', 'English', 'Biology', 'Political Science'],
  '9': ['Geography', 'History', 'Economics', 'English', 'Biology', 'Political Science'],
  '10': ['Geography', 'History', 'Economics', 'English', 'Biology', 'Political Science'],
};

export default function App() {
  const [step, setStep] = useState(3);
  const [selectedClass, setSelectedClass] = useState("");
  const [selectedSubject, setSelectedSubject] = useState("");
  const [generatedData, setGeneratedData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [showAnswers, setShowAnswers] = useState(false);
  const [questionCounts, setQuestionCounts] = useState({
    mcqs: 5,
    fill_in_the_blanks: 5,
    subjective: 5,
  });

  const nextStep = () => setStep((prev) => prev + 1);
  const prevStep = () => {
    setStep((prev) => prev - 1);
    setGeneratedData(null);
    setShowAnswers(false);
    setError("");
  };

  const createApiFormData = () => {
    if (!selectedClass || !selectedSubject) return null;
    const a = new FormData();
    a.append('class_num', selectedClass);
    a.append('subject', selectedSubject);
    return a;
  };
  
  const handleClassChange = (e) => {
    setSelectedClass(e.target.value);
    setSelectedSubject(""); 
  };

  const handleSubjectChange = (e) => {
    setSelectedSubject(e.target.value);
  };

  const handleCountChange = (e) => {
    const { name, value } = e.target;
    const a = Math.max(1, parseInt(value, 10) || 1);
    setQuestionCounts(prev => ({ ...prev, [name]: a }));
  };

  const generateQuestions = async () => {
    if (!selectedClass || !selectedSubject) return;
    setIsLoading(true);
    setError("");
    setGeneratedData(null);
    setShowAnswers(false);

    const a = new FormData();
    a.append('class_num', selectedClass);
    a.append('subject', selectedSubject);
    a.append('counts_json', JSON.stringify(questionCounts));

    try {
      const r = await fetch(`${API_URL}/generate-questions`, { method: "POST", body: a });
      if (!r.ok) {
        const e = await r.json();
        throw new Error(e.detail || "Failed to generate questions.");
      }
      const d = await r.json();
      setGeneratedData(d);
      nextStep();
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleShowAnswers = async () => {
    setIsLoading(true);
    setError("");
    
    const a = createApiFormData();
    if (!a) return;
    a.append('questions_json', JSON.stringify({ questions: generatedData }));

    try {
      const r = await fetch(`${API_URL}/generate-answers`, { method: "POST", body: a });
      if (!r.ok) {
        const e = await r.json();
        throw new Error(e.detail || "Failed to generate answers.");
      }
      const d = await r.json();
      setGeneratedData(d);
      setShowAnswers(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleEvaluate = async (qType, qIndex) => {
    const b = JSON.parse(JSON.stringify(generatedData));
    const c = b[qType][qIndex];
    c.evaluating = true;
    setGeneratedData(b);

    const a = createApiFormData();
    if (!a) return;
    a.append('question', c.question);
    a.append('generated_answer', c.answer);
    a.append('q_type', qType);

    try {
      const r = await fetch(`${API_URL}/evaluate-accuracy`, { method: 'POST', body: a });
      if (!r.ok) throw new Error('Evaluation failed.');
      const d = await r.json();
      c.score = d.similarity_score;
      c.bert_answer = d.bert_answer;
    } catch (e) {
      c.score = 'Error';
    } finally {
      c.evaluating = false;
      setGeneratedData(JSON.parse(JSON.stringify(b)));
    }
  };

  const handleHumanEvaluationChange = (qType, qIndex, value) => {
    const a = JSON.parse(JSON.stringify(generatedData));
    a[qType][qIndex].human_evaluation = value;
    setGeneratedData(a);
  };

  const handleRegenerateAnswer = async (qType, qIndex) => {
    const b = JSON.parse(JSON.stringify(generatedData));
    const c = b[qType][qIndex];
    c.regenerating = true;
    setGeneratedData(b);

    const a = createApiFormData();
    if (!a) return;
    a.append('question', c.question);
    a.append('original_answer', c.answer);
    a.append('human_evaluation', c.human_evaluation);
    a.append('bert_answer', c.bert_answer || 'N/A');
    a.append('q_type', qType); // Add this line

    try {
        const r = await fetch(`${API_URL}/regenerate-answer`, { method: 'POST', body: a });
        if (!r.ok) throw new Error('Regeneration failed.');
        const d = await r.json();
        c.answer = d.new_answer;
        c.score = null;
        c.bert_answer = null;
        c.human_evaluation = null;
    } catch (e) {
        setError("Failed to regenerate answer. Please try again.");
    } finally {
        c.regenerating = false;
        setGeneratedData(JSON.parse(JSON.stringify(b)));
    }
  };

  const downloadPDF = async () => {
    setIsLoading(true);
    try {
      const r = await fetch(`${API_URL}/answer-questions-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answered_questions: generatedData }),
      });
      if (!r.ok) throw new Error('PDF generation failed.');

      const d = await r.blob();
      const u = window.URL.createObjectURL(d);
      const e = document.createElement('a');
      e.href = u;
      e.download = "answers_script.pdf";
      document.body.appendChild(e);
      e.click();
      e.remove();
      window.URL.revokeObjectURL(u);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setStep(3);
    setSelectedClass("");
    setSelectedSubject("");
    setGeneratedData(null);
    setIsLoading(false);
    setError("");
    setShowAnswers(false);
  };
  
  const ScoreBadge = ({ score }) => {
    if (score === 'Error' || typeof score !== 'number') return <span className="text-red-500">Error</span>;
    const a = (score * 100).toFixed(1);
    let b = 'text-red-400';
    if (score > 0.8) b = 'text-green-400';
    else if (score > 0.6) b = 'text-yellow-400';
    return <span className={b}>Score: {a}%</span>;
  };

  return (
    <div className='flex flex-col items-center justify-center min-h-screen 
      bg-gradient-to-br from-black via-[#0f0b2d] to-purple-900 text-white p-6'>
      <AnimatePresence mode='wait'>
        {step === 3 && (
          <motion.div
            key='step3'
            initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -30 }}
            className='relative bg-[#1a1a2e]/90 backdrop-blur-xl rounded-2xl p-10 w-full max-w-3xl shadow-2xl border border-purple-800'>
            <h2 className='text-2xl font-bold mb-6 text-purple-300 flex items-center'>
              <BookOpen className='w-6 h-6 mr-2 text-purple-400' />
              Select Class and Subject
            </h2>
            <div className='w-full grid grid-cols-1 md:grid-cols-2 gap-6 mb-6'>
              <div>
                <label htmlFor="class-select" className="block text-sm font-medium text-purple-300 mb-2">Class</label>
                <select id="class-select" value={selectedClass} onChange={handleClassChange} className="w-full p-3 rounded-lg bg-[#0f0b2d] text-white border border-purple-700 focus:ring-2 focus:ring-purple-500 outline-none">
                  <option value="" disabled>Select a class</option>
                  {Object.keys(subjectOptions).map(c => <option key={c} value={c}>Class {c}</option>)}
                </select>
              </div>
              <div>
                <label htmlFor="subject-select" className="block text-sm font-medium text-purple-300 mb-2">Subject</label>
                <select id="subject-select" value={selectedSubject} onChange={handleSubjectChange} disabled={!selectedClass} className="w-full p-3 rounded-lg bg-[#0f0b2d] text-white border border-purple-700 focus:ring-2 focus:ring-purple-500 outline-none disabled:opacity-50">
                  <option value="" disabled>Select a subject</option>
                  {selectedClass && subjectOptions[selectedClass].map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
            </div>
            <div className="w-full grid grid-cols-3 gap-4 my-6">
              <div>
                <label htmlFor="mcqs" className="block text-sm font-medium text-purple-300 mb-2">MCQs</label>
                <input type="number" name="mcqs" id="mcqs" value={questionCounts.mcqs} onChange={handleCountChange} min="1" className="w-full p-2 rounded-lg bg-[#0f0b2d] text-white border border-purple-700 focus:ring-2 focus:ring-purple-500 outline-none text-center" />
              </div>
              <div>
                <label htmlFor="fill_in_the_blanks" className="block text-sm font-medium text-purple-300 mb-2">Fill in the Blanks</label>
                <input type="number" name="fill_in_the_blanks" id="fill_in_the_blanks" value={questionCounts.fill_in_the_blanks} onChange={handleCountChange} min="1" className="w-full p-2 rounded-lg bg-[#0f0b2d] text-white border border-purple-700 focus:ring-2 focus:ring-purple-500 outline-none text-center" />
              </div>
              <div>
                <label htmlFor="subjective" className="block text-sm font-medium text-purple-300 mb-2">Subjective</label>
                <input type="number" name="subjective" id="subjective" value={questionCounts.subjective} onChange={handleCountChange} min="1" className="w-full p-2 rounded-lg bg-[#0f0b2d] text-white border border-purple-700 focus:ring-2 focus:ring-purple-500 outline-none text-center" />
              </div>
            </div>
            {error && <p className="text-red-400 mt-4">{error}</p>}
            <div className='flex justify-end mt-6'>
              <button onClick={generateQuestions} disabled={!selectedClass || !selectedSubject || isLoading} className='px-6 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-lg shadow hover:opacity-90 disabled:opacity-50 transition flex items-center'>
                {isLoading && <Loader className="animate-spin w-4 h-4 mr-2" />}
                {isLoading ? "Generating..." : "Generate Questions →"}
              </button>
            </div>
          </motion.div>
        )}

        {step === 4 && (
          <motion.div
            key='step4'
            initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -30 }}
            className='relative bg-[#1a1a2e]/90 backdrop-blur-xl rounded-2xl p-10 w-full max-w-4xl shadow-2xl border border-purple-800'>
            <h2 className='text-2xl font-bold mb-6 text-indigo-300 flex items-center'>
              <MessageSquare className='w-6 h-6 mr-2 text-indigo-400' />
              Generated Content
            </h2>
              {error && <p className="text-red-400 mb-4">{error}</p>}
            <div className='bg-[#0f0b2d] text-white rounded-xl p-6 shadow-md 
                         border border-purple-700 h-[500px] overflow-y-auto text-left'>
              {generatedData && Object.keys(generatedData).map(qType => (
                <div key={qType} className="mb-6">
                  <h3 className="text-xl font-bold text-purple-400 mb-3 capitalize">{qType.replace(/_/g, ' ')}</h3>
                  <ul className="list-decimal list-inside space-y-6">
                    {generatedData[qType]?.map((item, i) => (
                      <li key={i} className="space-y-2">
                        <p className="whitespace-pre-line">{item.question}</p>
                        {showAnswers && item.answer && (
                          <div className="p-3 bg-indigo-900/50 rounded-md">
                            <p className="text-green-300 whitespace-pre-line">{item.answer}</p>
                            <div className="mt-4 pt-3 border-t border-purple-800/50 flex items-center justify-between gap-4 flex-wrap">
                              <div className="flex items-center gap-4">
                                {!item.score && !item.evaluating && (
                                  <button onClick={() => handleEvaluate(qType, i)} className="text-xs px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded">Check Accuracy</button>
                                )}
                                {item.evaluating && <Loader className="animate-spin w-4 h-4" />}
                                {item.score && <div className="text-sm"><p className="font-bold"><ScoreBadge score={item.score} /></p></div>}
                              </div>
                              <div className="flex items-center gap-4">
                                <select 
                                  value={item.human_evaluation || ""}
                                  onChange={(e) => handleHumanEvaluationChange(qType, i, e.target.value)}
                                  className="text-xs p-1 rounded bg-[#0f0b2d] border border-purple-700 text-white focus:ring-1 focus:ring-purple-500 outline-none"
                                >
                                  <option value="" disabled>Rate Answer</option>
                                  <option value="CORRECT">Correct</option>
                                  <option value="HALF_CORRECT">Half-Correct</option>
                                  <option value="FALSE">False</option>
                                </select>
                                {item.human_evaluation && item.human_evaluation !== 'CORRECT' && (
                                    !item.regenerating ? (
                                        <button onClick={() => handleRegenerateAnswer(qType, i)} className="text-xs px-2 py-1 bg-yellow-600 hover:bg-yellow-500 rounded flex items-center gap-1">
                                            <RefreshCw className="w-3 h-3"/>Regenerate
                                        </button>
                                    ) : (
                                        <Loader className="animate-spin w-4 h-4 text-yellow-400" />
                                    )
                                )}
                              </div>
                            </div>
                            {item.score && <p className="text-xs text-gray-400 mt-2">BERT Answer: "{item.bert_answer}"</p>}
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <div className='flex justify-between items-center mt-6'>
              <button onClick={prevStep} className='px-6 py-2 bg-gray-700 text-gray-200 rounded-lg hover:bg-gray-600'>← Back</button>
              {showAnswers ? (
                  <button disabled={isLoading} onClick={downloadPDF} className='px-6 py-2 bg-gradient-to-r from-green-500 to-teal-500 text-white rounded-lg shadow hover:opacity-90 transition flex items-center disabled:opacity-50'>{isLoading ? <Loader className="animate-spin w-4 h-4 mr-2" /> : <Download className="w-4 h-4 mr-2" />}Download PDF</button>
              ) : (
                  <button onClick={handleShowAnswers} disabled={isLoading} className='px-6 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-lg shadow hover:opacity-90 disabled:opacity-50 transition flex items-center'>{isLoading && <Loader className="animate-spin w-4 h-4 mr-2" />}Show Answers</button>
              )}
              <button onClick={resetForm} className='px-6 py-2 bg-gradient-to-r from-red-500 to-pink-500 text-white rounded-lg shadow hover:opacity-90 transition'>🔄 Start Over</button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}