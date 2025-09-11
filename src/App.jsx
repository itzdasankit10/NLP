import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileQuestion, MessageSquare, Loader, Download } from "lucide-react";

const API_URL = "http://localhost:8000";

export default function App() {
  const [step, setStep] = useState(3);
  const [context, setContext] = useState("");
  const [generatedData, setGeneratedData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [showAnswers, setShowAnswers] = useState(false);

  const nextStep = () => setStep((prev) => prev + 1);
  const prevStep = () => {
    setStep((prev) => prev - 1);
    setGeneratedData(null);
    setShowAnswers(false);
    setError("");
  };

  const generateQuestions = async () => {
    if (!context.trim()) return;
    setIsLoading(true);
    setError("");
    setGeneratedData(null);
    setShowAnswers(false);

    try {
      const res = await fetch(`${API_URL}/generate-questions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context: context }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to generate questions.");
      }
      const data = await res.json();
      setGeneratedData(data);
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
    try {
      const res = await fetch(`${API_URL}/generate-answers`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context: context, questions: generatedData }),
      });
       if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to generate answers.");
      }
      const data = await res.json();
      setGeneratedData(data);
      setShowAnswers(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };
// --- In App.jsx ---

// Find your handleEvaluate function and modify the `body` of the fetch request
const handleEvaluate = async (qType, qIndex) => {
    const newData = JSON.parse(JSON.stringify(generatedData));
    const item = newData[qType][qIndex];
    item.evaluating = true;
    setGeneratedData(newData);

    try {
      const res = await fetch(`${API_URL}/evaluate-accuracy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          context: context,
          question: item.question,
          generated_answer: item.answer,
          q_type: qType, // <<< ADD THIS LINE
        }),
      });
      // ... rest of the function is the same
      if (!res.ok) throw new Error('Evaluation failed.');
      const result = await res.json();
      item.score = result.similarity_score;
      item.bert_answer = result.bert_answer;

    } catch (e) {
        item.score = 'Error';
    } finally {
        item.evaluating = false;
        setGeneratedData(JSON.parse(JSON.stringify(newData)));
    }
  };

  const downloadPDF = async () => {
    setIsLoading(true);
    try {
        const res = await fetch(`${API_URL}/answer-questions-pdf`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ answered_questions: generatedData }),
        });
        if (!res.ok) throw new Error('PDF generation failed.');

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "answers_script.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (e) {
        setError(e.message);
    } finally {
        setIsLoading(false);
    }
  };

  const resetForm = () => {
    setStep(3);
    setContext("");
    setGeneratedData(null);
    setIsLoading(false);
    setError("");
    setShowAnswers(false);
  };
  
  const ScoreBadge = ({ score }) => {
    if (score === 'Error' || typeof score !== 'number') return <span className="text-red-500">Error</span>;
    const pct = (score * 100).toFixed(1);
    let color = 'text-red-400';
    if (score > 0.8) color = 'text-green-400';
    else if (score > 0.6) color = 'text-yellow-400';
    return <span className={color}>Score: {pct}%</span>
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
              <FileQuestion className='w-6 h-6 mr-2 text-purple-400' />
              Provide Context to Generate Questions
            </h2>
            <textarea
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder='Paste the text, article, or content here...'
              className='w-full h-[400px] p-4 rounded-lg bg-[#0f0b2d] text-white 
                       border border-purple-700 focus:ring-2 focus:ring-purple-500 outline-none'
            />
            {error && <p className="text-red-400 mt-4">{error}</p>}
            <div className='flex justify-end mt-6'>
              <button
                onClick={generateQuestions}
                disabled={!context || isLoading}
                className='px-6 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-lg shadow hover:opacity-90 disabled:opacity-50 transition flex items-center'>
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
                  <h3 className="text-xl font-bold text-purple-400 mb-3 capitalize">
                    {qType.replace(/_/g, ' ')}
                  </h3>
                  <ul className="list-decimal list-inside space-y-6">
                    {generatedData[qType]?.map((item, i) => (
                      <li key={i}>
                        <p className="whitespace-pre-line">{item.question}</p>
                        {showAnswers && item.answer && (
                          <div className="mt-2 p-3 bg-indigo-900/50 rounded-md">
                            <p className="text-green-300 whitespace-pre-line">{item.answer}</p>
                            {!item.score && !item.evaluating && (
                                <button onClick={() => handleEvaluate(qType, i)} className="mt-3 text-xs px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded">
                                    Check Accuracy
                                </button>
                            )}
                            {item.evaluating && <Loader className="animate-spin w-4 h-4 mt-3" />}
                            {item.score && (
                                <div className="mt-3 pt-3 border-t border-purple-800 text-sm">
                                    <p className="font-bold"><ScoreBadge score={item.score} /></p>
                                    <p className="text-gray-400 mt-1">BERT Answer: "{item.bert_answer}"</p>
                                </div>
                            )}
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