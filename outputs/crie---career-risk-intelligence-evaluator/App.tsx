import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { InputForm } from './components/InputForm';
import { AnalysisResultView } from './components/AnalysisResult';
import { InputData, AnalysisResult } from './types';
import { generateCareerAnalysis } from './services/geminiService';
import { GraduationCap, Globe } from 'lucide-react';

const App: React.FC = () => {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (data: InputData) => {
    setLoading(true);
    setError(null);
    try {
      const analysis = await generateCareerAnalysis(data);
      setResult(analysis);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "分析过程中发生错误，请稍后重试。如果问题持续，请检查 API Key。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pb-12">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 h-16 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="bg-primary/10 p-2 rounded-lg">
              <GraduationCap className="text-primary" size={24} />
            </div>
            <div>
              <h1 className="font-serif font-bold text-xl text-gray-900 leading-tight">CRIE</h1>
              <p className="text-[10px] text-gray-500 tracking-wider">CAREER RISK INTELLIGENCE</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-1 text-sm text-gray-500 cursor-pointer hover:text-primary transition-colors">
              <Globe size={16} />
              <span>中文 (CN)</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-3xl mx-auto px-4 pt-8">
        
        {/* Error Banner */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded-r">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Input Section - Collapsed if results present (could be optimized, keeping visible for now for refinement) */}
        {!result && (
           <div className="mb-8 text-center space-y-4 animate-fadeIn">
              <h1 className="text-3xl md:text-4xl font-serif font-bold text-gray-900 mt-8 mb-4">
                专业风险智能评估器
              </h1>
              <p className="text-gray-600 max-w-lg mx-auto leading-relaxed">
                利用大数据与AI模型，为您的高考志愿保驾护航。深度分析就业率、薪资前景及AI替代风险。
              </p>
           </div>
        )}

        <InputForm onSubmit={handleSearch} isLoading={loading} />

        {/* Results Section */}
        {result && <AnalysisResultView result={result} />}

      </main>
    </div>
  );
};

export default App;
