#!/usr/bin/env python3

import sqlite3
import numpy as np
import altair as alt
import sys
from scipy.spatial import ConvexHull
import os
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

DIR_ENVVAR = 'TOPK_DIR'
try:
    BASE_DIR = os.environ[DIR_ENVVAR]
except:
    print("You should set the {} environment variable to a directory".format(DIR_ENVVAR))
    sys.exit(1)

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RESULT_FILES_DIR = os.path.join(BASE_DIR, "output")


def get_db():
    db = sqlite3.connect(os.path.join(BASE_DIR, "join-results.db"))
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
        select dataset, workload, k, algorithm, algorithm_version, params, threads, json_extract(params, '$.hash_source') as hash_source, 
               recall, time_index_s, time_join_s, time_index_s + time_join_s as time_total_s 
        from recent 
         where json_extract(params, '$.prefix') is null
           and k = 10
           and workload = 'local-top-k';
        """, db)
    all = all.fillna(value={'hash_source': ''})
    all['algorithm'] = all['algorithm'] + all['hash_source']
    print(all)
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
        width=1000,
        height=600,
        title="Recall vs. time"
    ).add_selection(selection).interactive()
    chart.save(os.path.join(BASE_DIR, "plot.html"))


def plot_topk(workload):
    plotdir = os.path.join(BASE_DIR, "plots")
    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)

    db = get_db()
    all = pd.read_sql(f"""
        select dataset, workload, k, algorithm, algorithm_version, params, threads, json_extract(params, '$.method') as method, 
               recall, time_index_s, time_join_s, time_index_s + time_join_s as time_total_s 
        from recent 
         where json_extract(params, '$.prefix') is null
           and workload = '{workload}-top-k';
        """, db)
    all = all.fillna(value={'method': ''})
    all['algorithm'] = all['algorithm'] + all['method']
    print(all)
    data = get_pareto(all)

    # algorithms =[
    #     t[0]
    #     for t in db.execute(f"select distinct algorithm from recent where workload = '{workload}-top-k' order by 1;").fetchall()
    # ]
    algorithms = data["algorithm"].unique()
    colors = [
      "#5778a4",
      "#e49444",
      "#d1615d",
      "#85b6b2",
      "#6a9f58",
      "#e7ca60",
      "#a87c9f",
      "#f1a2a9",
      "#967662",
      "#b8b0ac"
    ]

    datasets = [
        t[0]
        for t in db.execute(f"select distinct dataset from recent where workload = '{workload}-top-k' order by 1;").fetchall()
    ]
    ks = [
        t[0]
        for t in db.execute(f"select distinct k from recent where workload = '{workload}-top-k' order by 1;").fetchall()
    ]
    color_mapping = alt.Color('algorithm', 
                              type='nominal', 
                              scale=alt.Scale(domain=algorithms, range=colors))

    k_radio = alt.binding_radio(options=ks, name='K: ')
    input_dropdown = alt.binding_select(options=datasets, name='Dataset: ')
    selection = alt.selection_single(fields=['dataset'], bind=input_dropdown, empty='none')
    k_selection = alt.selection_single(fields=['k'], bind=k_radio, empty='none')

    chart_pareto = alt.Chart(data).transform_filter(selection & k_selection).mark_line(point=True).encode(
        x=alt.X('recall', type='quantitative', scale=alt.Scale(domain=(0, 1))),
        y=alt.Y('time_total_s', type='quantitative', scale=alt.Scale(type='log')),
        color=color_mapping,
        tooltip=[
            'algorithm:N',
            'params:N',
            'recall:Q',
            'time_total_s:Q'
        ]
    )
    chart_all = alt.Chart(all).transform_filter(selection & k_selection).mark_point().encode(
        x=alt.X('recall', type='quantitative', scale=alt.Scale(domain=(0, 1))),
        y=alt.Y('time_total_s', type='quantitative', scale=alt.Scale(type='log')),
        color=color_mapping,
        tooltip=[
            'algorithm:N',
            'params:N',
            'recall:Q',
            'time_total_s:Q',
            'k:Q'
        ]
    )
    
    chart = alt.layer(chart_all, chart_pareto).properties(
        width=1000,
        height=600,
        title="Recall vs. time"
    ).add_selection(selection).add_selection(k_selection).interactive()
    chart.save(os.path.join(plotdir, f"plot-{workload}.html"))

    for dataset in datasets:
        for k in ks:
            plotdata = data[data['dataset'] == dataset]
            plotdata = plotdata[plotdata['k'] == k]
            if plotdata.shape[1] > 0:
                plt.figure()
                print(plotdata)
                sns.lineplot(
                    x = "recall",
                    y = "time_total_s",
                    hue = "algorithm",
                    palette = dict(zip(algorithms, colors)),
                    data=plotdata
                )
                sns.scatterplot(
                    x = "recall",
                    y = "time_total_s",
                    hue = "algorithm",
                    legend = False,
                    palette = dict(zip(algorithms, colors)),
                    data=plotdata
                )
                plt.yscale('log')
                plt.savefig(os.path.join(plotdir, f"plot-{workload}-{dataset}-k{k}.pdf"))



def plot_distance_histogram(path, k):
    plotdir = os.path.join(BASE_DIR, "plots")
    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)
    f = h5py.File(path)
    name = os.path.basename(path)
    kth_dist = f['top-1000-dists'][:,k-1]
    print(kth_dist)
    print("k=", k, "minimum similarity is", np.min(kth_dist))
    plt.figure()
    sns.kdeplot(kth_dist)
    plt.title("{} {}-nn distribution".format(name, k))
    outfile = os.path.join(plotdir, path + ".dists-k={}.pdf".format(k))
    print("saving to", outfile)
    plt.savefig(outfile)

def plot_distance_histograms(names, k):
    plotdir = os.path.join(BASE_DIR, "plots")
    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)
    alldata = []
    for name in names:
        path = run.DATASETS[name]()
        f = h5py.File(path)
        dists = f['top-1000-dists'][:,k-1]
        kth_dist = pd.DataFrame({'similarity': dists})
        kth_dist['dataset'] = name if name != 'glove-200' else 'Glove'
        alldata.append(kth_dist)
    data = pd.concat(alldata, ignore_index=True)
    print('plotting data with shape', data.shape)
    print(data['dataset'].unique())
    sns.displot(
        data = data,
        x    = 'similarity',
        kind = 'kde',
        bw_method = 0.2,
        cut = 0,
        fill =- True,
        col  = 'dataset',
        height = 2.5,
        aspect = 1,
        facet_kws = dict(sharey = False)
    )
    outfile = os.path.join(plotdir, "distances-at-k{}.png".format(k))
    print("saving to", outfile)
    plt.savefig(outfile, dpi=300)

def plot_param_dep():
    plotdir = os.path.join(BASE_DIR, "plots")
    with get_db() as db:
        data = pd.read_sql("""
            select dataset, workload, k, algorithm, algorithm_version, params, threads,
                   json_extract(params, '$.space_usage') * 1024.0 * 1024.0 as space_bytes, 
                   json_extract(params, '$.space_usage') * 1024.0 * 1024.0 / size as bytes_per_point, 
                   json_extract(params, '$.with_sketches') as with_sketches,
                   json_extract(params, '$.deduplicate') as deduplicate,
                   json_extract(params, '$.recall') as target_recall,
                   recall
                   time_index_s, time_join_s, time_index_s + time_join_s as time_total_s,
                   size / (time_index_s + time_join_s) as points_per_sec
            from recent natural join dataset_size
             where json_extract(params, '$.prefix') is null
               and algorithm = 'PUFFINN' 
               and dataset != 'movielens-1M'
               and k = 10
               and workload = 'local-top-k';
            """, db)
        data = data.fillna(value={'with_sketches': False, 'deduplicate': False})
    # print(data[['dataset', 'points_per_sec', 'bytes_per_point']])
    for target_recall in [0.8, 0.9]:
        data = data[data['target_recall'] == target_recall]
        sns.lineplot(
            x = 'bytes_per_point',
            y = 'points_per_sec',
            style = 'with_sketches',
            hue = 'dataset',
            markers = True,
            data = data
        )
        plt.title('Throughput vs. memory at target recall {}'.format(target_recall))
        plt.savefig(os.path.join(plotdir, 'qps-vs-mem-at-{}.png'.format(target_recall)))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        import run
        names = sys.argv[1].split(",")
        plot_distance_histograms(names, 10)
        # dataset_path = run.DATASETS[sys.argv[1]]()
        # for k in [1, 10, 100, 1000]:
        #     plot_distance_histogram(dataset_path, k)
    else:
        plot_param_dep()
        # plot_topk("global")
        plot_topk("local")

