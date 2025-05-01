import sys
import time
import numpy as np

def parse_args():
    # 解析命令行参数
    instance_file = None
    termination = 60
    seed = 1
    args = sys.argv[1:]
    i = 0
    vis = np.zeros(len(args), dtype=bool)
    while i < len(args):
        if args[i] == '-t':
            vis[i] = True
            vis[i+1] = True
            termination = int(args[i+1])
            i += 1
        elif args[i] == '-s':
            vis[i] = True
            vis[i+1] = True
            seed = int(args[i+1])
            i += 1
        i += 1
    i = 0
    while i < len(args):
        if not vis[i]:
            instance_file = args[i]
        i += 1
    return instance_file, termination, seed

def parse_instance(filename):
    # 解析CARP实例文件
    with open(filename, 'r') as f:
        lines = f.readlines()
    info = {}
    edges = []
    for line in lines:
        if ':' in line:
            key, val = line.split(':')
            info[key.strip()] = val.strip()
        elif line.strip().startswith('END'):
            break
        elif line.strip() and line[0].isdigit():
            parts = line.strip().split()
            u, v, cost, demand = map(int, parts)
            edges.append({'u': u, 'v': v, 'cost': cost, 'demand': demand})
    # 基本参数
    V = int(info['VERTICES'])
    depot = int(info['DEPOT'])
    required_edges = [e for e in edges if e['demand'] > 0]
    non_required_edges = [e for e in edges if e['demand'] == 0]
    vehicles = int(info['VEHICLES'])
    capacity = int(info['CAPACITY'])
    return {
        'V': V,
        'depot': depot,
        'edges': edges,
        'required_edges': required_edges,
        'non_required_edges': non_required_edges,
        'vehicles': vehicles,
        'capacity': capacity
    }

def build_graph(instance):
    # 构建邻接矩阵
    V = instance['V']
    INF = 1 << 30
    cost = np.full((V+1, V+1), INF, dtype=int)
    for e in instance['edges']:
        u, v, c = e['u'], e['v'], e['cost']
        cost[u][v] = c
        cost[v][u] = c
    for i in range(1, V+1):
        cost[i][i] = 0
    # Floyd-Warshall最短路
    for k in range(1, V+1):
        for i in range(1, V+1):
            for j in range(1, V+1):
                if cost[i][j] > cost[i][k] + cost[k][j]:
                    cost[i][j] = cost[i][k] + cost[k][j]
    return cost

def path_scanning(instance, cost, rng):
    # 基于Path Scanning的启发式初解
    required = set((min(e['u'], e['v']), max(e['u'], e['v'])) for e in instance['required_edges'])
    unserved = required.copy()
    routes = []
    depot = instance['depot']
    capacity = instance['capacity']
    while unserved:
        route = []
        load = 0
        curr = depot
        while True:
            candidates = []
            for e in instance['required_edges']:
                eid = (min(e['u'], e['v']), max(e['u'], e['v']))
                if eid in unserved and load + e['demand'] <= capacity:
                    # 选择最近的任务
                    dist = min(cost[curr][e['u']], cost[curr][e['v']])
                    candidates.append((dist, eid, e))
            if not candidates:
                break
            # 随机打乱，选择最近的
            rng.shuffle(candidates)
            candidates.sort(key=lambda x: x[0])
            _, eid, e = candidates[0]
            # 确定服务方向
            if cost[curr][e['u']] <= cost[curr][e['v']]:
                route.append((e['u'], e['v']))
                curr = e['v']
            else:
                route.append((e['v'], e['u']))
                curr = e['u']
            load += e['demand']
            unserved.remove(eid)
        routes.append(route)
    return routes

def calc_total_cost(routes, instance, cost):
    # 计算总成本
    edge_cost = {}
    for e in instance['edges']:
        edge_cost[(e['u'], e['v'])] = e['cost']
        edge_cost[(e['v'], e['u'])] = e['cost']
    depot = instance['depot']
    total = 0
    for route in routes:
        curr = depot
        for u, v in route:
            total += cost[curr][u]
            total += edge_cost[(u, v)]
            curr = v
        total += cost[curr][depot]
    return total

def format_solution(routes):
    # 格式化输出
    s = []
    for route in routes:
        s.append('0')
        for u, v in route:
            s.append(f'({u},{v})')
        s.append('0')
    return 's ' + ','.join(s)

def main():
    instance_file, termination, seed = parse_args()
    rng = np.random.default_rng(seed)
    instance = parse_instance(instance_file)
    cost = build_graph(instance)
    start = time.time()
    best_routes = path_scanning(instance, cost, rng)
    best_cost = calc_total_cost(best_routes, instance, cost)
    # 时间控制（如需迭代改进，可在此循环）
    while time.time() - start < termination - 1:
        # 可加入局部搜索或多次path scanning取最优
        routes = path_scanning(instance, cost, rng)
        total_cost = calc_total_cost(routes, instance, cost)
        if total_cost < best_cost:
            best_cost = total_cost
            best_routes = routes
    print(format_solution(best_routes))
    print(f'q {best_cost}')

if __name__ == '__main__':
    main()