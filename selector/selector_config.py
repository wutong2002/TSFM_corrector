# selector/selector_config.py
# 说明：用来统一管理各种 selector, 加载信息和结果文件命名规则。

Selector_zoo_details = {
    "Random_Select": {
        # selector 类所在模块：from selector.baseline_select import Random_Select_Model
        "model_module": "selector.baseline_select",
        # selector 类名
        "model_class": "Random_Select_Model",
        # 结果文件命名模板
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}_seed{seed}_"
            "all_results.csv"
        ),
    },

    "All_Select": {
        "model_module": "selector.baseline_select",
        "model_class": "All_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}_all_ensemble_"
            "all_results.csv"
        ),
    },

    "Real_Select": {
        "model_module": "selector.baseline_select",
        "model_class": "Real_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}_real_{real_order_metric}_"
            "all_results.csv"
        ),
    },

    "Recent_Select": {
        "model_module": "selector.baseline_select",
        "model_class": "Recent_Select_Model",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}_recent_"
            "all_results.csv"
        ),
    },

    "MyFancy_Select": {
        "model_module": "selector.my_fancy_select",
        "model_class": "MyFancySelectModel",
        "csv_name_tpl": (
            "zoo{current_zoo_num}-{zoo_total_num}_"
            "top{ensemble_size}_myfancy_"
            "all_results.csv"
        ),
    },
}
