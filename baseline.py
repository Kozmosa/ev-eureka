import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class PremiumPredictionModel:
    """
    根据中国机动车商业保险示范产品基准纯风险保费表(2020版)的公式结构预测保费。
    由于缺乏实际的费率表和具体系数，本模型将使用基于输入数据特征的估算值和假设。
    """

    def __init__(self, additional_expense_rate=0.25,
                 traffic_violation_coeff=1.0,
                 underwriting_coeff=1.0,
                 channel_coeff=1.0):
        """
        初始化模型参数。
        :param additional_expense_rate: 附加费用率 (例如 0.25 表示 25%)
        :param traffic_violation_coeff: 交通违法系数
        :param underwriting_coeff: 自主核保系数
        :param channel_coeff: 自主渠道系数
        """
        self.additional_expense_rate = additional_expense_rate
        self.traffic_violation_coeff = traffic_violation_coeff
        self.underwriting_coeff = underwriting_coeff
        self.channel_coeff = channel_coeff

    def _estimate_standard_vehicle_value(self, make, type_vehicle, prod_year, effective_yr, ccm_ton, insured_value_agreed):
        """
        估算车辆的“新车购置价减去折旧金额后”的标准实际价值。
        这是一个非常简化的估算，实际中会使用精确的车型数据和折旧表。
        """
        vehicle_age = max(0, effective_yr - prod_year)
        
        # 粗略估计新车价格 (基于类型和排量/吨位，实际应查表)
        base_new_price = 100000  # 基础价格
        if type_vehicle == "轿车":
            base_new_price = 80000 + ccm_ton * 30
        elif type_vehicle == "SUV":
            base_new_price = 120000 + ccm_ton * 40
        elif type_vehicle == "MPV":
            base_new_price = 100000 + ccm_ton * 35
        elif type_vehicle == "货车":
            base_new_price = 150000 + ccm_ton * 20 # ccm_ton here is actually carrying_capacity for trucks in this mock
        elif type_vehicle == "客车":
            base_new_price = 200000 + ccm_ton * 15 # ccm_ton for buses
        
        # 简单年限折旧 (例如每年10%)
        depreciation_rate_annual = 0.10
        standard_depreciated_value = base_new_price * ((1 - depreciation_rate_annual) ** vehicle_age)
        
        # 使其与insured_value_agreed不要偏离太远，模拟一个“合理”的标准价值
        # 在实际情况中，这个值是独立于协商价值的
        # For this simulation, let's make it somewhat related to insured_value_agreed to avoid extreme adjustments
        # This is a major simplification.
        # A more robust way would be to have a proper new car price database.
        # Here, we'll return a value that could be, for example, 90-110% of insured_value_agreed,
        # or more realistically, derived independently as above.
        # To make the formula work, let's assume our estimated standard_depreciated_value is the one to use.
        # If it's too far from insured_value_agreed, the adjustment term will be large.
        return max(10000, standard_depreciated_value) # Ensure a minimum value


    def _get_mock_vehicle_loss_bprp_direct_lookup(self, usage, type_vehicle, vehicle_age, standard_depreciated_value):
        """
        模拟查询机动车损失保险的“直接查询基准纯风险保费”。
        这个值应基于车辆使用性质、种类、使用年限和标准实际价值。
        实际中这将是一个复杂的查表过程。
        """
        base_rate = 0.008 # 基础费率因子，纯属假设

        if type_vehicle == "轿车":
            base_rate = 0.01
        elif type_vehicle == "SUV":
            base_rate = 0.012
        elif type_vehicle == "MPV":
            base_rate = 0.011
        elif type_vehicle == "货车":
            base_rate = 0.015
        elif type_vehicle == "客车":
            base_rate = 0.018
        
        age_factor = 1 + (vehicle_age * 0.05) # 年龄越大风险越高（简化）
        usage_factor = 1.0
        if usage == "商用": usage_factor = 1.5
        elif usage == "租赁": usage_factor = 1.8
        
        # 估算的BPRP = 标准价值 * 基础费率 * 年龄因子 * 用途因子
        bprp = standard_depreciated_value * base_rate * age_factor * usage_factor
        return max(100, bprp) # 假设最低保费

    def _calculate_vehicle_loss_bprp(self, data_row):
        """计算考虑实际价值差异的机动车损失保险基准纯风险保费"""
        effective_yr = data_row['EFFECTIVE_YR']
        prod_year = data_row['PROD_YEAR']
        vehicle_age = max(0, effective_yr - prod_year)
        
        insured_value_agreed = data_row['INSURED_VALUE'] # 协商确定的机动车实际价值

        # 估算“新车购置价减去折旧金额后的机动车实际价值”
        standard_depreciated_value = self._estimate_standard_vehicle_value(
            data_row['MAKE'], data_row['TYPE_VEHICLE'], prod_year, effective_yr, data_row['CCM_TON'], insured_value_agreed
        )
        
        # 直接查询的机动车损失保险基准纯风险保费 (基于标准价值)
        bprp_direct_lookup = self._get_mock_vehicle_loss_bprp_direct_lookup(
            data_row['USAGE'], data_row['TYPE_VEHICLE'], vehicle_age, standard_depreciated_value
        )
        
        # 考虑实际价值差异的调整
        bprp_adjusted_for_value_diff = bprp_direct_lookup + \
            (insured_value_agreed - standard_depreciated_value) * 0.0009 # 0.09% = 0.0009
            
        # 假设没有约定绝对免赔额，因此费率折扣系数为1
        # final_bprp_loss = bprp_adjusted_for_value_diff *费率折扣系数 (默认为1)
        final_bprp_loss = bprp_adjusted_for_value_diff
        return final_bprp_loss

    def _calculate_third_party_bprp(self, data_row):
        """计算第三者责任保险基准纯风险保费"""
        if not data_row.get('insurance_commercial_third_party_active', 0): # 检查TPL是否激活
            return 0

        usage = data_row['USAGE']
        type_vehicle = data_row['TYPE_VEHICLE']
        liability_limit = data_row.get('third_party_liability_limit', 1000000) # 从数据行获取，若无则默认

        # 模拟查表获取BPRP
        base_bprp = 500 # 基础值
        if type_vehicle == "轿车": base_bprp = 400 + liability_limit * 0.0005
        elif type_vehicle == "SUV": base_bprp = 450 + liability_limit * 0.00055
        elif type_vehicle == "MPV": base_bprp = 420 + liability_limit * 0.00052
        elif type_vehicle == "货车": base_bprp = 700 + liability_limit * 0.0008
        elif type_vehicle == "客车": base_bprp = 800 + liability_limit * 0.001
        
        usage_factor = 1.0
        if usage == "商用": usage_factor = 1.8
        elif usage == "租赁": usage_factor = 2.2
        
        return max(50, base_bprp * usage_factor) # 假设最低保费

    def _get_mock_personnel_risk_rate(self, usage, type_vehicle, personnel_type="driver"):
        """模拟查询车上人员责任险的纯风险费率"""
        base_rate = 0.001 # 驾驶员基础费率
        if personnel_type == "passenger":
            base_rate = 0.0005 # 乘客基础费率
        
        if type_vehicle == "货车": base_rate *= 1.5
        if type_vehicle == "客车": base_rate *= 2.0
        if usage == "商用": base_rate *= 1.5
        if usage == "租赁": base_rate *= 1.8
        return base_rate

    def _calculate_personnel_liability_bprp(self, data_row):
        """计算车上人员责任保险基准纯风险保费"""
        if not data_row.get('include_personnel_liability', False):
            return 0

        usage = data_row['USAGE']
        type_vehicle = data_row['TYPE_VEHICLE']
        
        # 驾驶人
        driver_limit = data_row.get('driver_liability_limit', 50000)
        driver_risk_rate = self._get_mock_personnel_risk_rate(usage, type_vehicle, "driver")
        bprp_driver = driver_limit * driver_risk_rate
        
        # 乘客
        passenger_limit_per_seat = data_row.get('passenger_liability_limit_per_seat', 10000)
        # 确保SEATS_NUM至少为1（驾驶员座位），乘客座位数为SEATS_NUM - 1 (如果SEATS_NUM > 0)
        # 但公式是“投保乘客座位数”，这里假设SEATS_NUM就是总的投保座位数（含驾驶员，或者就是乘客座位数，看如何定义）
        # 通常车上人员责任险分开买司机和乘客。这里简化：SEATS_NUM如果是总座位数，乘客座位数为 SEATS_NUM -1 (假设至少有一个司机)
        # 如果SEATS_NUM已经是乘客座位数，则直接用。这里假设SEATS_NUM是包含驾驶员的总座位数。
        # 实践中，乘客座位数是单独指定的。
        num_passenger_seats_insured = max(0, data_row['SEATS_NUM'] -1) # 假设驾驶员单独算，其余为乘客
        if num_passenger_seats_insured == 0 and data_row['SEATS_NUM'] > 0 : # e.g. single seater or data means total seats
             # If SEATS_NUM is 1 (likely driver), or if it's intended that all seats can be passengers
             # For simplicity, let's assume SEATS_NUM can be used as a proxy for insured passenger seats if > 1
             # Or, if SEATS_NUM is the number of *passenger* seats specified, use it directly.
             # The prompt: "投保乘客座位数". Let's assume SEATS_NUM is this value for passengers if > 1.
             # If SEATS_NUM is 2, it could mean 1 driver + 1 passenger.
             # This part is ambiguous without clearer data definition for SEATS_NUM in context of this insurance.
             # Let's assume SEATS_NUM in data is total seats. If SEATS_NUM > 1, then (SEATS_NUM - 1) are passenger seats.
             # If SEATS_NUM = 1, passenger_bprp = 0.
             # If the problem means SEATS_NUM is specifically passenger seats, then the logic would change.
             # Given the data generation, SEATS_NUM is total seats.
            pass # num_passenger_seats_insured is already max(0, data_row['SEATS_NUM'] -1)

        passenger_risk_rate = self._get_mock_personnel_risk_rate(usage, type_vehicle, "passenger")
        bprp_passengers = passenger_limit_per_seat * passenger_risk_rate * num_passenger_seats_insured
        
        return bprp_driver + bprp_passengers

    def _calculate_ncd_factor(self, claim_paid):
        """计算无赔款优待系数 (NCD) - 简化版"""
        # 实际NCD系数由行业平台根据历史赔款记录返回，并有详细的档位
        # 此处为极简模拟：有赔款则系数 > 1，无赔款则 < 1
        if claim_paid > 0:
            return 1.1  # 有赔款记录，费率上浮10% (示例)
        else:
            return 0.85 # 无赔款记录，费率下浮15% (示例)

    def predict_premium(self, data_row):
        """
        预测单条记录的商业车险保费。
        :param data_row: Pandas Series, 代表一行车辆和保险数据。
        :return: float, 预测的商业车险保费。
        """
        # 1. 计算各险种基准纯风险保费之和
        bprp_loss = self._calculate_vehicle_loss_bprp(data_row)
        bprp_tpl = self._calculate_third_party_bprp(data_row)
        bprp_personnel = self._calculate_personnel_liability_bprp(data_row)
        
        total_bprp = bprp_loss + bprp_tpl + bprp_personnel
        
        if total_bprp == 0: # 如果没有任何险种激活或计算为0
            return 0

        # 2. 计算基准保费
        # 基准保费 = 基准纯风险保费 / (1 - 附加费用率)
        if (1 - self.additional_expense_rate) == 0:
            # Avoid division by zero, though additional_expense_rate should not be 1
            raise ValueError("Additional expense rate cannot be 100%.")
        base_premium = total_bprp / (1 - self.additional_expense_rate)
        
        # 3. 计算费率调整系数
        # 无赔款优待系数
        ncd_coeff = self._calculate_ncd_factor(data_row['CLAIM_PAID'])
        # 交通违法系数 (使用初始化时的默认值或传入值)
        traffic_coeff = self.traffic_violation_coeff 
        # 自主核保系数 (使用初始化时的默认值或传入值)
        underwriting_coeff = self.underwriting_coeff
        # 自主渠道系数 (使用初始化时的默认值或传入值)
        channel_coeff = self.channel_coeff
        
        rate_adjustment_factor = ncd_coeff * traffic_coeff * underwriting_coeff * channel_coeff
        
        # 4. 计算商业车险总保费
        commercial_premium = base_premium * rate_adjustment_factor
        
        return round(commercial_premium, 2)

# 主程序示例
if __name__ == "__main__":
    # 生成10条样本数据用于演示
    df_vehicle_data = pd.read_csv('test.csv')
    
    print(f"已生成{len(df_vehicle_data)}条样本数据。")
    print("\n部分生成数据预览 (前5条):")
    print(df_vehicle_data.head().to_string())
    print("-" * 50)

    # 初始化保费预测模型
    # 可以传入自定义的系数，例如：
    # model = PremiumPredictionModel(additional_expense_rate=0.30, traffic_violation_coeff=1.1)
    model = PremiumPredictionModel() 
    
    # 对每条数据进行保费预测
    predicted_premiums = df_vehicle_data.apply(lambda row: model.predict_premium(row), axis=1)
    df_vehicle_data['predicted_premium'] = predicted_premiums
    
    print("\n保费预测结果 (部分字段及预测保费):")
    columns_to_show = ['OBJECT_ID', 'TYPE_VEHICLE', 'USAGE', 'INSURED_VALUE', 
                       'CLAIM_PAID', 'PREMIUM', 'predicted_premium']
    print(df_vehicle_data[columns_to_show].head().to_string())
    df_vehicle_data.to_csv("baseline_predicted.csv")

    # 详细展示一条记录的计算过程（可选）
    if not df_vehicle_data.empty:
        print("\n详细计算示例 (第一条记录):")
        sample_row = df_vehicle_data.iloc[0]
        print(f"  输入数据: {sample_row[columns_to_show].to_dict()}")
        bprp_loss = model._calculate_vehicle_loss_bprp(sample_row)
        bprp_tpl = model._calculate_third_party_bprp(sample_row)
        bprp_personnel = model._calculate_personnel_liability_bprp(sample_row)
        total_bprp = bprp_loss + bprp_tpl + bprp_personnel
        base_premium_calc = total_bprp / (1 - model.additional_expense_rate)
        ncd = model._calculate_ncd_factor(sample_row['CLAIM_PAID'])
        adjustment_factor = ncd * model.traffic_violation_coeff * model.underwriting_coeff * model.channel_coeff

        print(f"  车损险BPRP: {bprp_loss:.2f}")
        print(f"  三者险BPRP: {bprp_tpl:.2f}")
        print(f"  车上人员BPRP: {bprp_personnel:.2f}")
        print(f"  总BPRP: {total_bprp:.2f}")
        print(f"  附加费用率: {model.additional_expense_rate}")
        print(f"  基准保费: {base_premium_calc:.2f}")
        print(f"  NCD系数: {ncd:.2f}")
        print(f"  总费率调整系数: {adjustment_factor:.2f}")
        print(f"  最终预测商业险保费: {sample_row['predicted_premium']:.2f}")