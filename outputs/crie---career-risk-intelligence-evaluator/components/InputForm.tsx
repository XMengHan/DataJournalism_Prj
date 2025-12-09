import React, { useState } from 'react';
import { InputData } from '../types';
import { Search, Loader2 } from 'lucide-react';

interface Props {
  onSubmit: (data: InputData) => void;
  isLoading: boolean;
}

const CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "其他"];
const STREAMS = ["理科/物理类", "文科/历史类", "综合"];
const FAMILIES = ["城市中产", "农村家庭", "高收入家庭", "不限"];

export const InputForm: React.FC<Props> = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState<InputData>({
    gaokao_score: 550,
    target_city: "上海",
    arts_science: "理科/物理类",
    family_situation: "不限",
    risk_preference: "平衡型"
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6 md:p-8 mb-8">
      <div className="mb-6 border-b border-gray-100 pb-4">
        <h2 className="text-xl font-serif font-bold text-primary flex items-center gap-2">
          <span className="text-2xl">📝</span> 请输入您的基本信息
        </h2>
        <p className="text-gray-500 text-sm mt-1">CRIE 将基于这些信息为您生成精准的职业风险评估报告。</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Score */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">高考分数 (200-750)</label>
            <input
              type="number"
              min="200"
              max="750"
              value={formData.gaokao_score}
              onChange={(e) => setFormData({...formData, gaokao_score: parseInt(e.target.value)})}
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-primary focus:border-transparent transition-all outline-none text-lg font-bold text-primary"
              required
            />
          </div>

          {/* City */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">意向城市</label>
            <select
              value={formData.target_city}
              onChange={(e) => setFormData({...formData, target_city: e.target.value})}
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-primary focus:border-transparent transition-all outline-none bg-white"
            >
              {CITIES.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          {/* Stream */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">文理分科</label>
            <div className="flex gap-2">
              {STREAMS.map(s => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setFormData({...formData, arts_science: s})}
                  className={`flex-1 py-2 px-2 text-sm rounded-lg border transition-all ${
                    formData.arts_science === s 
                      ? 'bg-primary text-white border-primary' 
                      : 'bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

           {/* Family - Optional/Advanced */}
           <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">家庭经济背景 (影响建议)</label>
             <select
              value={formData.family_situation}
              onChange={(e) => setFormData({...formData, family_situation: e.target.value})}
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-primary focus:border-transparent transition-all outline-none bg-white"
            >
              {FAMILIES.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
        </div>

        <div className="pt-4">
          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-4 rounded-xl text-white font-bold text-lg shadow-lg transition-all transform hover:scale-[1.01] active:scale-[0.99] flex justify-center items-center gap-2 ${
              isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-primary hover:bg-blue-600'
            }`}
          >
            {isLoading ? (
              <>
                <Loader2 className="animate-spin" />
                正在智能分析大数据...
              </>
            ) : (
              <>
                <Search size={20} />
                开始智能分析 🚀
              </>
            )}
          </button>
          <p className="text-center text-xs text-gray-400 mt-3">
            * 分析过程约需 5-10 秒，请耐心等待
          </p>
        </div>
      </form>
    </div>
  );
};
