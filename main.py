import argparse
import logging
from src.data_loader import load_data, generate_fitness_data
from src.analysis import FitnessAnalyzer
from src.utils import chunked_csv_reader, low_sleep_generator


def cmd_generate(args):
    generate_fitness_data(args.output)


def cmd_analyze(args):
    df = load_data(args.input)
    analyzer = FitnessAnalyzer(df)
    print("\n=== ОПИСАТЕЛЬНАЯ СТАТИСТИКА ===")
    print(analyzer.basic_stats())


def cmd_stats(args):
    df = load_data(args.input)
    analyzer = FitnessAnalyzer(df)
    print("\n=== СОН: БУДНИ vs ВЫХОДНЫЕ ===")
    print(analyzer.weekday_vs_weekend_sleep())
    print("\n=== КОРРЕЛЯЦИЯ ШАГИ-СОН ===")
    print(f"Коэффициент корреляции: {analyzer.steps_sleep_correlation():.3f}")
    print("\n=== САМЫЙ АКТИВНЫЙ ДЕНЬ ===")
    print(analyzer.most_active_day())
    print("\n=== СОН ПО СЕЗОНАМ ===")
    print(analyzer.sleep_season_trend())
    print("\n=== ГЛУБОКИЙ СОН И ПУЛЬС ===")
    print(f"Корреляция deep_sleep и resting_heart_rate: {analyzer.deep_sleep_vs_heart_rate():.3f}")


def cmd_report(args):
    df = load_data(args.input)
    analyzer = FitnessAnalyzer(df)

    print("\n========== ПОЛНЫЙ ОТЧЁТ ==========")
    # 1. Сон будни/выходные
    sleep_comp = analyzer.weekday_vs_weekend_sleep()
    print("\n1. Сон в будни vs выходные:")
    print(sleep_comp)
    print("Вывод: В выходные дни средняя продолжительность сна выше на "
          f"{sleep_comp.loc['Выходные','mean'] - sleep_comp.loc['Будни','mean']:.2f} ч.\n")

    # 2. Корреляция шаги-сон
    corr = analyzer.steps_sleep_correlation()
    print(f"2. Связь физической активности и сна: корреляция r = {corr:.3f}")
    if corr > 0.1:
        print("Вывод: Наблюдается слабая положительная связь — больше шагов -> немного дольше сон.")
    else:
        print("Вывод: Значимой линейной связи не выявлено.\n")

    # 3. Самый активный день
    best_day = analyzer.most_active_day()
    print(f"3. Самый активный день недели: {best_day}")
    print("Вывод: В этот день в среднем наибольшее количество шагов.\n")

    # 4. Сезонность сна
    seasons = analyzer.sleep_season_trend()
    print("4. Сон по сезонам:")
    print(seasons)
    print(f"Вывод: Зимой сон в среднем на {seasons.loc['winter','mean'] - seasons.loc['spring','mean']:.2f} ч длиннее, чем весной.\n")

    # 5. Глубокий сон и пульс
    hr_corr = analyzer.deep_sleep_vs_heart_rate()
    print(f"5. Влияние глубокого сна на пульс: корреляция r = {hr_corr:.3f}")
    print("Вывод: Чем больше глубокого сна, тем ниже пульс покоя (отрицательная корреляция).\n")

    # Демонстрация генераторов
    print("=== ДЕМОНСТРАЦИЯ ГЕНЕРАТОРОВ ===")
    print("Первые 2 чанка (по 5 записей):")
    chunk_gen = chunked_csv_reader(args.input, chunksize=5)
    for i, chunk in enumerate(chunk_gen):
        if i >= 2:
            break
        print(chunk[["date", "sleep_hours", "steps"]].to_string(index=False), "\n")

    print("Дни с недосыпом (<6 ч):")
    low_gen = low_sleep_generator(df, threshold=6.0)
    for i, row in enumerate(low_gen):
        print(f"{row['date'].strftime('%Y-%m-%d')} | {row['day_of_week']} | сон {row['sleep_hours']} ч")
        if i >= 4:
            break


def main():
    parser = argparse.ArgumentParser(description="Анализ сна и активности по данным фитнес-браслета")
    subparsers = parser.add_subparsers(title="Команды", dest="command")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Сгенерировать тестовые данные")
    gen_parser.add_argument("-o", "--output", default="data/fitness_data.csv", help="Путь для сохранения CSV")
    gen_parser.set_defaults(func=cmd_generate)

    # analyze
    an_parser = subparsers.add_parser("analyze", help="Базовая статистика")
    an_parser.add_argument("-i", "--input", default="data/fitness_data.csv", help="Входной CSV")
    an_parser.set_defaults(func=cmd_analyze)

    # stats
    st_parser = subparsers.add_parser("stats", help="Детальная статистика и зависимости")
    st_parser.add_argument("-i", "--input", default="data/fitness_data.csv", help="Входной CSV")
    st_parser.set_defaults(func=cmd_stats)

    # report
    rep_parser = subparsers.add_parser("report", help="Полный отчёт с выводами")
    rep_parser.add_argument("-i", "--input", default="data/fitness_data.csv", help="Входной CSV")
    rep_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        args.func(args)


if __name__ == "__main__":
    main()