import React from 'react';
import { AnalysisResult as ResultType } from '../types';
import { MajorCard } from './MajorCard';
import { RiskAnalysis } from './RiskAnalysis';
// Fix: Added Briefcase to imports
import { Download, Share2, Sparkles, BookOpen, TrendingUp, Lightbulb, Briefcase } from 'lucide-react';

interface Props {
  result: ResultType;
}

export const AnalysisResultView: React.FC<Props> = ({ result }) => {
  return (
    <div className="animate-slideUp fade-in-up">
      
      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 text-center">
          <div className="text-xs text-gray-500 mb-1">匹配专业数</div>
          <div className="text-2xl font-bold text-primary">{result.analysis_summary.total_matched_majors}</div>
        </div>
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 text-center">
          <div className="text-xs text-gray-500 mb-1">平均就业率</div>
          <div className="text-2xl font-bold text-gray-800">{result.analysis_summary.avg_employment_rate}%</div>
        </div>
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 text-center">
           <div className="text-xs text-gray-500 mb-1">平均抗AI指数</div>
           <div className={`text-2xl font-bold ${result.analysis_summary.avg_ai_risk_index > 70 ? 'text-success' : 'text-warning'}`}>
             {result.analysis_summary.avg_ai_risk_index}
           </div>
        </div>
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 text-center">
           <div className="text-xs text-gray-500 mb-1">置信度</div>
           <div className="text-2xl font-bold text-gray-800">{(result.analysis_summary.recommendation_confidence * 100).toFixed(0)}%</div>
        </div>
      </div>

      <div className="flex items-center gap-2 mb-6">
         <Sparkles className="text-primary" />
         <h2 className="text-xl font-bold text-gray-800">智能推荐 Top {result.top_recommendations.length}</h2>
      </div>

      {/* Recommendations List */}
      <div>
        {result.top_recommendations.map((major, index) => (
          <MajorCard key={index} data={major} index={index} />
        ))}
      </div>

      {/* Risk Analysis Section */}
      <RiskAnalysis 
        highRisk={result.risk_analysis.high_risk_majors}
        overrated={result.risk_analysis.overrated_majors}
      />

      {/* Personalized Advice */}
      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 md:p-8 mb-8 border border-blue-100">
        <h3 className="text-lg font-bold text-blue-900 mb-4 flex items-center gap-2">
           <Lightbulb size={20} /> 专家级个性化建议
        </h3>
        
        <div className="space-y-4">
          <div className="bg-white/60 p-4 rounded-lg">
             <div className="text-sm font-bold text-blue-800 mb-1">总体策略</div>
             <p className="text-gray-700 text-sm leading-relaxed">{result.personalized_advice.overall_strategy}</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
             <div className="bg-white/60 p-4 rounded-lg">
               <div className="flex items-center gap-2 text-sm font-bold text-gray-800 mb-2">
                  <BookOpen size={16} /> 考研建议
               </div>
               <p className="text-sm text-gray-600">{result.personalized_advice.graduate_school_advice}</p>
             </div>
             <div className="bg-white/60 p-4 rounded-lg">
               <div className="flex items-center gap-2 text-sm font-bold text-gray-800 mb-2">
                  <Briefcase size={16} /> 考公建议
               </div>
               <p className="text-sm text-gray-600">{result.personalized_advice.civil_service_advice}</p>
             </div>
          </div>

           <div className="bg-white/60 p-4 rounded-lg">
             <div className="flex items-center gap-2 text-sm font-bold text-gray-800 mb-2">
                <TrendingUp size={16} /> 技能树规划
             </div>
             <div className="flex flex-wrap gap-2">
               {result.personalized_advice.skill_development.map((skill, i) => (
                 <span key={i} className="px-2 py-1 bg-white border border-gray-200 rounded text-xs text-gray-700 shadow-sm">
                   {skill}
                 </span>
               ))}
             </div>
          </div>
        </div>
      </div>

      {/* Footer / Actions */}
      <div className="flex flex-col md:flex-row gap-4 justify-between items-center bg-gray-900 text-white p-6 rounded-xl">
        <div className="text-sm text-gray-400">
          <p className="mb-1">{result.disclaimer}</p>
          <p className="text-xs">数据来源: {result.data_sources.slice(0, 3).join(", ")} 等</p>
        </div>
        <div className="flex gap-3 shrink-0">
          <button className="flex items-center gap-2 px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors text-sm">
            <Share2 size={16} /> 分享
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-primary rounded-lg hover:bg-blue-600 transition-colors text-sm font-bold">
            <Download size={16} /> 下载完整报告
          </button>
        </div>
      </div>

    </div>
  );
};