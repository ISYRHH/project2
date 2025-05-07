import sys
import time
import numpy as np

def parse_args():
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
    V = instance['V']
    INF = 1 << 30
    cost = np.full((V+1, V+1), INF, dtype=int)
    for e in instance['edges']:
        u, v, c = e['u'], e['v'], e['cost']
        cost[u][v] = c
        cost[v][u] = c
    for i in range(1, V+1):
        cost[i][i] = 0
    for k in range(1, V+1):
        for i in range(1, V+1):
            for j in range(1, V+1):
                if cost[i][j] > cost[i][k] + cost[k][j]:
                    cost[i][j] = cost[i][k] + cost[k][j]
    return cost

def path_scanning(instance, cost, rng):
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
                    dist = min(cost[curr][e['u']], cost[curr][e['v']])
                    candidates.append((dist, eid, e))
            if not candidates:
                break
            rng.shuffle(candidates)
            candidates.sort(key=lambda x: x[0])
            _, eid, e = candidates[0]
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

def calc_route_cost(route, instance, cost):
    edge_cost = {}
    for e in instance['edges']:
        edge_cost[(e['u'], e['v'])] = e['cost']
        edge_cost[(e['v'], e['u'])] = e['cost']
    depot = instance['depot']
    total = 0
    curr = depot
    for u, v in route:
        total += cost[curr][u]
        total += edge_cost[(u, v)]
        curr = v
    total += cost[curr][depot]
    return total

def format_solution(routes):
    s = []
    for route in routes:
        s.append('0')
        for u, v in route:
            s.append(f'({u},{v})')
        s.append('0')
    return 's ' + ','.join(s)

def best_modify(new_route, ord_route, instance, cost):
    depot = instance['depot']
    rev_route = [(v, u) for u, v in ord_route[::-1]]
    s, t = ord_route[0][0], ord_route[-1][1]
    min_cost = cost[depot][s] + cost[t][new_route[0][0]] - cost[depot][new_route[0][0]]
    min_pos = 0
    min_rev = False
    if cost[depot][t] + cost[s][new_route[0][0]] - cost[depot][new_route[0][0]] < min_cost:
        min_cost = cost[depot][t] + cost[s][new_route[0][0]] - cost[depot][new_route[0][0]]
        min_pos = 0
        min_rev = True
    if cost[depot][s] + cost[t][new_route[-1][1]] - cost[depot][new_route[-1][1]] < min_cost:
        min_cost = cost[depot][s] + cost[t][new_route[-1][1]] - cost[depot][new_route[-1][1]]
        min_pos = len(new_route)
        min_rev = True
    if cost[depot][t] + cost[s][new_route[-1][1]] - cost[depot][new_route[-1][1]] < min_cost:
        min_cost = cost[depot][t] + cost[s][new_route[-1][1]] - cost[depot][new_route[-1][1]]
        min_pos = len(new_route)
        min_rev = False
    for i in range(1, len(new_route)):
        if cost[new_route[i-1][1]][s] + cost[new_route[i][0]][t] - cost[new_route[i-1][1]][new_route[i][0]] < min_cost:
            min_cost = cost[new_route[i-1][1]][s] + cost[new_route[i][0]][t] - cost[new_route[i-1][1]][new_route[i][0]]
            min_pos = i
            min_rev = False
        if cost[new_route[i][0]][s] + cost[new_route[i-1][1]][t] - cost[new_route[i][0]][new_route[i-1][1]] < min_cost:
            min_cost = cost[new_route[i][0]][s] + cost[new_route[i-1][1]][t] - cost[new_route[i][0]][new_route[i-1][1]]
            min_pos = i
            min_rev = True
    if min_rev:
        new_route = new_route[:min_pos] + rev_route + new_route[min_pos:]
    else:
        new_route = new_route[:min_pos] + ord_route + new_route[min_pos:]

    return new_route, calc_route_cost(new_route, instance, cost)

def modify(routes, ccost, instance, cost, rng):
    x = rng.integers(len(routes))
    if len (routes[x]) < 2:
        return routes, ccost
    l, r = rng.choice(len(routes[x]), size=2, replace=True)
    if l > r:
        l, r = r, l
    while r-l+1 == len(routes[x]):
        l, r = rng.choice(len(routes[x]), size=2, replace=True)
        if l > r:
            l, r = r, l
    new_route, new_cost = best_modify(routes[x][:l] + routes[x][r+1:], routes[x][l:r+1], instance, cost)
    return routes[:x] + [new_route] + routes[x+1:], ccost + new_cost - calc_route_cost(routes[x], instance, cost)

def main():
    instance_file, termination, seed = parse_args()
    rng = np.random.default_rng(seed)
    instance = parse_instance(instance_file)
    cost = build_graph(instance)
    start = time.time()
    best_routes = path_scanning(instance, cost, rng)
    best_cost = calc_total_cost(best_routes, instance, cost)
    max_iter = instance['V'] >> 2
    while time.time() - start < termination - 1:
        curr_routes = path_scanning(instance, cost, rng)
        curr_cost = calc_total_cost(curr_routes, instance, cost)
        for i in range(max_iter):
            curr_routes, curr_cost = modify(curr_routes, curr_cost, instance, cost, rng)
            # print(curr_cost)
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_routes = curr_routes
            if time.time() - start > termination - 1:
                break
    print(format_solution(best_routes))
    print(f'q {best_cost}')

if __name__ == '__main__':
    main()