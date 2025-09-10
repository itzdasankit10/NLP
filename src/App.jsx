//--------- full code with all four pages ---------------

// import React, { useState } from "react";
// import { motion, AnimatePresence } from "framer-motion";
// import {
//   BookOpen,
//   Globe,
//   Landmark,
//   TrendingUp,
//   GraduationCap,
//   FileQuestion,
//   MessageSquare
// } from "lucide-react";

// const subjects = [
//   { name: "History", icon: <Landmark className='inline w-5 h-5 mr-2' /> },
//   { name: "Geography", icon: <Globe className='inline w-5 h-5 mr-2' /> },
//   { name: "Economics", icon: <TrendingUp className='inline w-5 h-5 mr-2' /> },
//   {
//     name: "Political Science",
//     icon: <BookOpen className='inline w-5 h-5 mr-2' />
//   }
// ];

// const classes = ["Class 6", "Class 7", "Class 8", "Class 9"];

// export default function App() {
//   const [step, setStep] = useState(1);
//   const [selectedSubject, setSelectedSubject] = useState("");
//   const [selectedClass, setSelectedClass] = useState("");
//   const [question, setQuestion] = useState("");
//   const [answer, setAnswer] = useState("");

//   const nextStep = () => setStep((prev) => prev + 1);
//   const prevStep = () => setStep((prev) => prev - 1);

//   const generateAnswer = () => {
//     setAnswer(
//       `Here is a generated answer for "${question}" in ${selectedSubject}, ${selectedClass}.\n\nThis is just placeholder text, but the box is now EXTRA LARGE so you can see long answers easily.`
//     );
//     nextStep();
//   };

//   const resetForm = () => {
//     setStep(1);
//     setSelectedSubject("");
//     setSelectedClass("");
//     setQuestion("");
//     setAnswer("");
//   };

//   return (
//     <div
//       className='flex flex-col items-center justify-center min-h-screen 
//       bg-gradient-to-br from-black via-[#0f0b2d] to-purple-900 text-white p-6'>
//       {/* Stepper */}
//       <div className='flex items-center mb-10'>
//         {["Subject", "Class", "Question", "Answer"].map((label, index) => {
//           const isActive = step === index + 1;
//           return (
//             <div key={index} className='flex items-center'>
//               <div
//                 className={`flex items-center justify-center w-10 h-10 rounded-full text-white shadow-md
//                   ${
//                     isActive
//                       ? "bg-gradient-to-r from-purple-500 to-indigo-500"
//                       : "bg-gray-600"
//                   }`}>
//                 {index + 1}
//               </div>
//               <span
//                 className={`ml-2 mr-4 font-medium ${
//                   isActive ? "text-purple-400" : "text-gray-400"
//                 }`}>
//                 {label}
//               </span>
//               {index < 3 && <div className='w-10 h-[2px] bg-gray-600'></div>}
//             </div>
//           );
//         })}
//       </div>

//       {/* Steps with animation */}
//       <AnimatePresence mode='wait'>
//         {/* Step 1: Subject */}
//         {step === 1 && (
//           <motion.div
//             key='step1'
//             initial={{ opacity: 0, y: 30 }}
//             animate={{ opacity: 1, y: 0 }}
//             exit={{ opacity: 0, y: -30 }}
//             transition={{ duration: 0.4 }}
//             className='backdrop-blur-xl bg-[#1a1a2e]/80 shadow-lg rounded-2xl p-6 w-[36rem] border border-purple-800'>
//             <h2 className='text-xl font-bold mb-4 flex items-center text-purple-300'>
//               <BookOpen className='w-5 h-5 mr-2 text-purple-400' />
//               Select Subject
//             </h2>
//             <select
//               value={selectedSubject}
//               onChange={(e) => setSelectedSubject(e.target.value)}
//               className='w-full p-3 border rounded-lg bg-[#0f0b2d] text-white border-purple-700 focus:ring-2 focus:ring-purple-500 outline-none'>
//               <option value=''>-- Choose Subject --</option>
//               {subjects.map((subject, idx) => (
//                 <option
//                   key={idx}
//                   value={subject.name}
//                   className='bg-[#0f0b2d] text-white'>
//                   {subject.name}
//                 </option>
//               ))}
//             </select>
//             <div className='flex justify-end mt-6'>
//               <button
//                 onClick={nextStep}
//                 disabled={!selectedSubject}
//                 className='px-6 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-lg shadow hover:opacity-90 disabled:opacity-50 transition'>
//                 Next →
//               </button>
//             </div>
//           </motion.div>
//         )}

//         {/* Step 2: Class */}
//         {step === 2 && (
//           <motion.div
//             key='step2'
//             initial={{ opacity: 0, y: 30 }}
//             animate={{ opacity: 1, y: 0 }}
//             exit={{ opacity: 0, y: -30 }}
//             transition={{ duration: 0.4 }}
//             className='backdrop-blur-xl bg-[#1a1a2e]/80 shadow-lg rounded-2xl p-6 w-[36rem] border border-purple-800'>
//             <h2 className='text-xl font-bold mb-4 flex items-center text-purple-300'>
//               <GraduationCap className='w-5 h-5 mr-2 text-purple-400' />
//               Select Class
//             </h2>
//             <div className='grid grid-cols-2 gap-3'>
//               {classes.map((cls, idx) => (
//                 <button
//                   key={idx}
//                   onClick={() => setSelectedClass(cls)}
//                   className={`p-3 border rounded-lg transition 
//                     ${
//                       selectedClass === cls
//                         ? "bg-gradient-to-r from-purple-500 to-indigo-500 text-white shadow-md"
//                         : "bg-[#0f0b2d] text-gray-200 hover:bg-purple-900"
//                     }`}>
//                   {cls}
//                 </button>
//               ))}
//             </div>
//             <div className='flex justify-between mt-6'>
//               <button
//                 onClick={prevStep}
//                 className='px-6 py-2 bg-gray-700 text-gray-200 rounded-lg hover:bg-gray-600'>
//                 ← Back
//               </button>
//               <button
//                 onClick={nextStep}
//                 disabled={!selectedClass}
//                 className='px-6 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-lg shadow hover:opacity-90 disabled:opacity-50 transition'>
//                 Next →
//               </button>
//             </div>
//           </motion.div>
//         )}

//         {/* Step 3: Question Box (Bigger) */}
//         {step === 3 && (
//           <motion.div
//             key='step3'
//             initial={{ opacity: 0, y: 30 }}
//             animate={{ opacity: 1, y: 0 }}
//             exit={{ opacity: 0, y: -30 }}
//             transition={{ duration: 0.4 }}
//             className='relative bg-[#1a1a2e]/90 backdrop-blur-xl rounded-2xl p-10 w-[48rem] shadow-2xl border border-purple-800'>
//             <h2 className='text-2xl font-bold mb-6 text-purple-300 flex items-center'>
//               <FileQuestion className='w-6 h-6 mr-2 text-purple-400' />
//               Question Generate Box
//             </h2>
//             <textarea
//               value={question}
//               onChange={(e) => setQuestion(e.target.value)}
//               placeholder='Type your question here...'
//               className='w-full h-[500px] p-4 rounded-lg bg-[#0f0b2d] text-white 
//                          border border-purple-700 focus:ring-2 focus:ring-purple-500 outline-none'
//             />
//             <div className='flex justify-between mt-6'>
//               <button
//                 onClick={prevStep}
//                 className='px-6 py-2 bg-gray-700 text-gray-200 rounded-lg hover:bg-gray-600'>
//                 ← Back
//               </button>
//               <button
//                 onClick={generateAnswer}
//                 disabled={!question}
//                 className='px-6 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-lg shadow hover:opacity-90 disabled:opacity-50 transition'>
//                 Generate Answer →
//               </button>
//             </div>
//           </motion.div>
//         )}

//         {/* Step 4: Answer Box (Bigger) */}
//         {step === 4 && (
//           <motion.div
//             key='step4'
//             initial={{ opacity: 0, y: 30 }}
//             animate={{ opacity: 1, y: 0 }}
//             exit={{ opacity: 0, y: -30 }}
//             transition={{ duration: 0.4 }}
//             className='relative bg-[#1a1a2e]/90 backdrop-blur-xl rounded-2xl p-10 w-[48rem] shadow-2xl border border-purple-800'>
//             <h2 className='text-2xl font-bold mb-6 text-indigo-300 flex items-center'>
//               <MessageSquare className='w-6 h-6 mr-2 text-indigo-400' />
//               Answer Generate Box
//             </h2>
//             <div
//               className='bg-[#0f0b2d] text-white rounded-xl p-6 shadow-md 
//                             border border-purple-700 h-[500px] overflow-y-auto'>
//               <p className='text-lg whitespace-pre-line'>{answer}</p>
//             </div>
//             <div className='flex justify-between mt-6'>
//               <button
//                 onClick={prevStep}
//                 className='px-6 py-2 bg-gray-700 text-gray-200 rounded-lg hover:bg-gray-600'>
//                 ← Back
//               </button>
//               <button
//                 onClick={resetForm}
//                 className='px-6 py-2 bg-gradient-to-r from-red-500 to-pink-500 text-white rounded-lg shadow hover:opacity-90 transition'>
//                 🔄 Start Over
//               </button>
//             </div>
//           </motion.div>
//         )}
//       </AnimatePresence>
//     </div>
//   );
// }


//--------- only last two pages ---------------

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileQuestion, MessageSquare } from "lucide-react";

export default function App() {
  const [step, setStep] = useState(3); // ⬅️ start directly from Step 3
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const nextStep = () => setStep((prev) => prev + 1);
  const prevStep = () => setStep((prev) => prev - 1);

  const generateAnswer = () => {
    setAnswer(
      `Here is a generated answer for your question:\n\n"${question}"\n\nThis is placeholder text, but the box is now EXTRA LARGE so you can see long answers easily.`
    );
    nextStep();
  };

  const resetForm = () => {
    setStep(3); // reset back to Question Box
    setQuestion("");
    setAnswer("");
  };

  return (
    <div
      className='flex flex-col items-center justify-center min-h-screen 
      bg-gradient-to-br from-black via-[#0f0b2d] to-purple-900 text-white p-6'>
      {/* Stepper */}
      <div className='flex items-center mb-10'>
        {["Question", "Answer"].map((label, index) => {
          const isActive = step === index + 3; // step 3 & 4 only
          return (
            <div key={index} className='flex items-center'>
              <div
                className={`flex items-center justify-center w-10 h-10 rounded-full text-white shadow-md
                  ${
                    isActive
                      ? "bg-gradient-to-r from-purple-500 to-indigo-500"
                      : "bg-gray-600"
                  }`}>
                {index + 1}
              </div>
              <span
                className={`ml-2 mr-4 font-medium ${
                  isActive ? "text-purple-400" : "text-gray-400"
                }`}>
                {label}
              </span>
              {index < 1 && <div className='w-10 h-[2px] bg-gray-600'></div>}
            </div>
          );
        })}
      </div>

      {/* Steps */}
      <AnimatePresence mode='wait'>
        {/* Step 3: Question Box */}
        {step === 3 && (
          <motion.div
            key='step3'
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.4 }}
            className='relative bg-[#1a1a2e]/90 backdrop-blur-xl rounded-2xl p-10 w-[48rem] shadow-2xl border border-purple-800'>
            <h2 className='text-2xl font-bold mb-6 text-purple-300 flex items-center'>
              <FileQuestion className='w-6 h-6 mr-2 text-purple-400' />
              Question Generate Box
            </h2>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder='Type your question here...'
              className='w-full h-[500px] p-4 rounded-lg bg-[#0f0b2d] text-white 
                         border border-purple-700 focus:ring-2 focus:ring-purple-500 outline-none'
            />
            <div className='flex justify-end mt-6'>
              <button
                onClick={generateAnswer}
                disabled={!question}
                className='px-6 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-lg shadow hover:opacity-90 disabled:opacity-50 transition'>
                Generate Answer →
              </button>
            </div>
          </motion.div>
        )}

        {/* Step 4: Answer Box */}
        {step === 4 && (
          <motion.div
            key='step4'
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.4 }}
            className='relative bg-[#1a1a2e]/90 backdrop-blur-xl rounded-2xl p-10 w-[48rem] shadow-2xl border border-purple-800'>
            <h2 className='text-2xl font-bold mb-6 text-indigo-300 flex items-center'>
              <MessageSquare className='w-6 h-6 mr-2 text-indigo-400' />
              Answer Generate Box
            </h2>
            <div
              className='bg-[#0f0b2d] text-white rounded-xl p-6 shadow-md 
                            border border-purple-700 h-[500px] overflow-y-auto'>
              <p className='text-lg whitespace-pre-line'>{answer}</p>
            </div>
            <div className='flex justify-between mt-6'>
              <button
                onClick={prevStep}
                className='px-6 py-2 bg-gray-700 text-gray-200 rounded-lg hover:bg-gray-600'>
                ← Back
              </button>
              <button
                onClick={resetForm}
                className='px-6 py-2 bg-gradient-to-r from-red-500 to-pink-500 text-white rounded-lg shadow hover:opacity-90 transition'>
                🔄 Start Over
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
