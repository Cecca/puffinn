import sqlite3
import numpy as np
import altair as alt
import sys
import os
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns


def get_db():
    db = sqlite3.connect("join-results.db")
    return db


def get_pareto(data):
    def compute_pareto(gdata):
        gdata = gdata.sort_values(['time_total_s'], ascending=True)
        points = np.vstack(
            (gdata['recall'], gdata['time_total_s'])
        ).transpose()

        # now we seek the vertices of the pareto 
        # frontier to select from the `gdata` object
        indices = []
        last_r = 0
        for i, (r, t) in enumerate(points):
            if r > last_r:
                last_r = r
                indices.append(i)
        return gdata[['recall', 'time_total_s', 'params']].iloc[indices]

    # data = pd.read_sql("select dataset, workload, k, algorithm, params, threads, recall, time_index_s, time_join_s, time_index_s + time_join_s as time_total_s from main;", get_db())
    pareto = data.groupby(['dataset', 'workload', 'k', 'algorithm', 'threads']).apply(compute_pareto)
    return pareto.reset_index()


def plot_local_topk():
    db = get_db()
    all = pd.read_sql("""
        select 
            dataset, 
            k, 
            CASE
                WHEN json_extract(params, '$.method') = 'LSH' THEN 'PUFFINN-query'
                WHEN json_extract(params, '$.method') = 'LSHJoin' THEN 'PUFFINN-join'
                ELSE algorithm
            END AS algorithm, 
            params, 
            threads, 
            json_extract(params, '$.hash_source') as hash_source, 
               recall, time_index_s, time_join_s, time_index_s + time_join_s as time_total_s 
        from recent 
         where json_extract(params, '$.prefix') is null
           and dataset in (
             'glove-200',
             'DeepImage',
             'Orkut',
             'DBLP'
           )
           and k = 10
           and workload = 'local-top-k';
        """, db)
    data = get_pareto(all)

    datasets = [
        t[0]
        for t in db.execute("select distinct dataset from recent order by 1;").fetchall()
    ]

    input_dropdown = alt.binding_select(options=datasets, name='Dataset: ')
    selection = alt.selection_single(fields=['dataset'], bind=input_dropdown, empty='none')

    chart_pareto = alt.Chart(data).transform_filter(selection).mark_line(point=True).encode(
        x=alt.X('recall', type='quantitative', scale=alt.Scale(domain=(0, 1))),
        y=alt.Y('time_total_s', type='quantitative', scale=alt.Scale(type='log')),
        color='algorithm:N',
        tooltip=[
            'algorithm:N',
            'params:N',
            'recall:Q',
            'time_total_s:Q'
        ]
    )
    chart_all = alt.Chart(all).transform_filter(selection).mark_point().encode(
        x=alt.X('recall', type='quantitative', scale=alt.Scale(domain=(0, 1))),
        y=alt.Y('time_total_s', type='quantitative', scale=alt.Scale(type='log')),
        color='algorithm:N',
        tooltip=[
            'algorithm:N',
            'params:N',
            'recall:Q',
            'time_total_s:Q'
        ]
    )
    
    chart = alt.layer(chart_all, chart_pareto).properties(
        # width=1000,
        # height=600,
        title="Recall vs. time"
    ).add_selection(selection).interactive()
    return chart


# def plot_topk(workload):
#     plotdir = os.path.join(BASE_DIR, "plots")
#     if not os.path.isdir(plotdir):
#         os.mkdir(plotdir)
#
#     db = get_db()
#     all = pd.read_sql(f"""
#         select dataset, workload, k, algorithm, algorithm_version, params, threads, json_extract(params, '$.method') as method, 
#                recall, time_index_s, time_join_s, time_index_s + time_join_s as time_total_s 
#         from recent 
#          where json_extract(params, '$.prefix') is null
#            and workload = '{workload}-top-k';
#         """, db)
#     all = all.fillna(value={'method': ''})
#     all['algorithm'] = all['algorithm'] + all['method']
#     data = get_pareto(all)
#
#     # algorithms =[
#     #     t[0]
#     #     for t in db.execute(f"select distinct algorithm from recent where workload = '{workload}-top-k' order by 1;").fetchall()
#     # ]
#     algorithms = data["algorithm"].unique()
#     colors = [
#       "#5778a4",
#       "#e49444",
#       "#d1615d",
#       "#85b6b2",
#       "#6a9f58",
#       "#e7ca60",
#       "#a87c9f",
#       "#f1a2a9",
#       "#967662",
#       "#b8b0ac"
#     ]
#
#     datasets = [
#         t[0]
#         for t in db.execute(f"select distinct dataset from recent where workload = '{workload}-top-k' order by 1;").fetchall()
#     ]
#     ks = [
#         t[0]
#         for t in db.execute(f"select distinct k from recent where workload = '{workload}-top-k' order by 1;").fetchall()
#     ]
#     color_mapping = alt.Color('algorithm', 
#                               type='nominal', 
#                               scale=alt.Scale(domain=algorithms, range=colors))
#
#     k_radio = alt.binding_radio(options=ks, name='K: ')
#     input_dropdown = alt.binding_select(options=datasets, name='Dataset: ')
#     selection = alt.selection_single(fields=['dataset'], bind=input_dropdown, empty='none')
#     k_selection = alt.selection_single(fields=['k'], bind=k_radio, empty='none')
#
#     chart_pareto = alt.Chart(data).transform_filter(selection & k_selection).mark_line(point=True).encode(
#         x=alt.X('recall', type='quantitative', scale=alt.Scale(domain=(0, 1))),
#         y=alt.Y('time_total_s', type='quantitative', scale=alt.Scale(type='log')),
#         color=color_mapping,
#         tooltip=[
#             'algorithm:N',
#             'params:N',
#             'recall:Q',
#             'time_total_s:Q'
#         ]
#     )
#     chart_all = alt.Chart(all).transform_filter(selection & k_selection).mark_point().encode(
#         x=alt.X('recall', type='quantitative', scale=alt.Scale(domain=(0, 1))),
#         y=alt.Y('time_total_s', type='quantitative', scale=alt.Scale(type='log')),
#         color=color_mapping,
#         tooltip=[
#             'algorithm:N',
#             'params:N',
#             'recall:Q',
#             'time_total_s:Q',
#             'k:Q'
#         ]
#     )
#     
#     chart = alt.layer(chart_all, chart_pareto).properties(
#         width=1000,
#         height=600,
#         title="Recall vs. time"
#     ).add_selection(selection).add_selection(k_selection).interactive()
#     chart.save(os.path.join(plotdir, f"plot-{workload}.html"))
#
#     for dataset in datasets:
#         for k in ks:
#             plotdata = data[data['dataset'] == dataset]
#             plotdata = plotdata[plotdata['k'] == k]
#             if plotdata.shape[1] > 0:
#                 plt.figure()
#                 sns.lineplot(
#                     x = "recall",
#                     y = "time_total_s",
#                     hue = "algorithm",
#                     palette = dict(zip(algorithms, colors)),
#                     data=plotdata
#                 )
#                 sns.scatterplot(
#                     x = "recall",
#                     y = "time_total_s",
#                     hue = "algorithm",
#                     legend = False,
#                     palette = dict(zip(algorithms, colors)),
#                     data=plotdata
#                 )
#                 plt.yscale('log')
#                 plt.savefig(os.path.join(plotdir, f"plot-{workload}-{dataset}-k{k}.pdf"))
#
#
#
#
