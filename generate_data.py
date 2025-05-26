import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

# 初始化 Faker 用于生成随机数据
fake = Faker('zh_CN')

def generate_vehicle_data(num_samples):
    """生成车辆保险与驾驶行为模拟数据"""
    data = []
    
    # 定义可能的品牌列表
    brands = ["品牌A", "品牌B", "品牌C", "品牌D", "品牌E"]
    car_makes = ["大众", "丰田", "本田", "别克", "奥迪", "宝马", "奔驰", "比亚迪", "吉利", "长安"]
    vehicle_types = ["轿车", "SUV", "MPV", "皮卡", "货车", "客车"]
    usages = ["家用", "商用", "租赁", "公务", "自用货物"]
    insurance_types = [1201, 1202, 1203, 1204, 1205, 1206]  # 不同保险类型代码
    sex_types = [0, 1]  # 0: 男, 1: 女
    
    for i in range(num_samples):
        # 基础信息
        object_id = 5000000000 + i
        sex = random.choice(sex_types)
        brand = random.choice(brands)
        make = random.choice(car_makes)
        type_vehicle = random.choice(vehicle_types)
        usage = random.choice(usages)
        insurance_type = random.choice(insurance_types)
        
        # 时间相关
        start_date = fake.date_between(start_date='-3y', end_date='-1y')
        end_date = start_date + timedelta(days=365)
        effective_yr = int(start_date.year)
        prod_year = random.randint(effective_yr - 10, effective_yr)
        
        # 车辆属性
        if type_vehicle in ["轿车", "SUV", "MPV"]:
            seats_num = random.choice([4, 5, 7])
            carrying_capacity = 0  # 乘用车载重量为0
            ccm_ton = round(random.uniform(1.0, 3.5), 1) * 1000  # 排量(cc)
        else:  # 货车、皮卡
            seats_num = random.choice([2, 3, 4])
            carrying_capacity = random.randint(1, 20)  # 载重量(吨)
            ccm_ton = round(random.uniform(2.0, 6.0), 1) * 1000  # 排量(cc)
        
        # 保险相关
        insured_value = round(random.uniform(50000, 500000), 2)
        premium = round(insured_value * random.uniform(0.005, 0.03), 2)
        insurance_commercial = random.randint(0, 1)
        insurance_compulsory = 1  # 交强险通常为1
        
        # 驾驶行为相关
        average_speed = round(random.uniform(30, 100), 2)
        avg_daily_charges = round(random.uniform(10, 100), 2) if brand in ["品牌C", "品牌D"] else 0
        fatigue_driving_ratio = round(random.uniform(0, 0.2), 6)
        late_night_trip_ratio = round(random.uniform(0, 0.3), 6)
        avg_late_night_trip_mileage = round(random.uniform(0, 10), 2) if late_night_trip_ratio > 0 else 0
        high_temp_driving_ratio = round(random.uniform(0, 0.2), 6)
        
        # 电池相关 (电动车)
        is_electric = random.randint(0, 1)
        battery_type_lfp = is_electric * random.randint(0, 1)
        initial_battery_soc = round(random.uniform(30, 90), 2) if is_electric else 0
        avg_charge_duration = round(random.uniform(1800, 7200), 2) if is_electric else 0
        
        # 理赔相关 (约30%的样本有理赔)
        has_claim = random.random() < 0.3
        claim_paid = round(random.uniform(1000, 50000), 2) if has_claim else 0
        average_loss = round(random.uniform(500, 20000), 2) if has_claim else 0
        
        # 添加到数据列表
        data.append({
            'SEX': sex,
            'INSR_BEGIN': start_date.strftime('%d-%b-%y').upper(),
            'INSR_END': end_date.strftime('%d-%b-%y').upper(),
            'EFFECTIVE_YR': effective_yr,
            'INSR_TYPE': insurance_type,
            'INSURED_VALUE': insured_value,
            'PREMIUM': premium,
            'OBJECT_ID': object_id,
            'PROD_YEAR': prod_year,
            'SEATS_NUM': seats_num,
            'CARRYING_CAPACITY': carrying_capacity,
            'TYPE_VEHICLE': type_vehicle,
            'CCM_TON': ccm_ton,
            'MAKE': make,
            'USAGE': usage,
            'CLAIM_PAID': claim_paid,
            'brand': brand,
            'average_speed': average_speed,
            'avg_daily_charges': avg_daily_charges,
            'fatigue_driving_ratio': fatigue_driving_ratio,
            'late_night_trip_ratio': late_night_trip_ratio,
            'avg_late_night_trip_mileage': avg_late_night_trip_mileage,
            'high_temp_driving_ratio': high_temp_driving_ratio,
            'battery_type_lfp': battery_type_lfp,
            'initial_battery_soc': initial_battery_soc,
            'avg_charge_duration': avg_charge_duration,
            'insurance_commercial_third_party': insurance_commercial,
            'insurance_compulsory_third_party': insurance_compulsory,
            'average_loss': average_loss
        })
    
    return pd.DataFrame(data)

# 生成样本数据
if __name__ == "__main__":
    # 生成1000条样本数据
    df = generate_vehicle_data(1000)
    
    # 保存为CSV文件
    df.to_csv('vehicle_insurance_data.csv', index=False)
    
    print(f"已生成{len(df)}条样本数据并保存至 vehicle_insurance_data.csv")
    
    # 显示数据前几行
    print("\n数据前几行预览:")
    print(df.head().to_csv(sep='\t', na_rep='nan'))