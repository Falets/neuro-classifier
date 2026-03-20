import os
import pandas as pd
import argparse

def split_csv_into_three_equal_parts(file_path, output_dir):
    df = pd.read_csv(file_path)
    total_rows = len(df)
    part_size = total_rows // 3
    remainder = total_rows % 3

    split1 = part_size + (1 if remainder > 0 else 0)
    split2 = part_size * 2 + (1 if remainder > 1 else 0)

    part1 = df.iloc[:split1]
    part2 = df.iloc[split1:split2]
    part3 = df.iloc[split2:]

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    part_folder = os.path.join(output_dir, base_name)
    os.makedirs(part_folder, exist_ok=True)

    part1.to_csv(os.path.join(part_folder, f"{base_name}_part1.csv"), index=False)
    part2.to_csv(os.path.join(part_folder, f"{base_name}_part2.csv"), index=False)
    part3.to_csv(os.path.join(part_folder, f"{base_name}_part3.csv"), index=False)

    print(f"{base_name}: {len(part1)} | {len(part2)} | {len(part3)} строк")

def main():
    parser = argparse.ArgumentParser(description='Разделить каждый CSV на три равные части, сохранить в подпапки с именами частей, включающими исходное имя.')
    parser.add_argument('input_folder', help='Папка с исходными CSV')
    parser.add_argument('output_folder', help='Папка для сохранения')
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Ошибка: папка {args.input_folder} не найдена")
        return

    os.makedirs(args.output_folder, exist_ok=True)

    for file in os.listdir(args.input_folder):
        if file.endswith('.csv'):
            split_csv_into_three_equal_parts(os.path.join(args.input_folder, file), args.output_folder)

    print("Готово!")

if __name__ == '__main__':
    main()