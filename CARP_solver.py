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
    edge_demand = {}
    for e in edges:
        edge_demand[(e['u'], e['v'])] = e['demand']
        edge_demand[(e['v'], e['u'])] = e['demand']
    return {
        'V': V,
        'depot': depot,
        'edges': edges,
        'required_edges': required_edges,
        'non_required_edges': non_required_edges,
        'vehicles': vehicles,
        'capacity': capacity,
        'edge_demand': edge_demand
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
    depot = instance['depot']
    total = 0
    curr = depot
    for u, v in route:
        total += cost[curr][u]
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

def best_modify(routex, routey, instance, cost):
    if len(routex) == 0 or len(routey) == 0:
        return routex + routey, calc_route_cost(routex + routey, instance, cost)
    depot = instance['depot']
    rev_route = [(v, u) for u, v in routey[::-1]]
    s, t = routey[0][0], routey[-1][1]
    min_cost = cost[depot][s] + cost[t][routex[0][0]] - cost[depot][routex[0][0]]
    min_pos = 0
    min_rev = False
    if cost[depot][t] + cost[s][routex[0][0]] - cost[depot][routex[0][0]] < min_cost:
        min_cost = cost[depot][t] + cost[s][routex[0][0]] - cost[depot][routex[0][0]]
        min_pos = 0
        min_rev = True
    if cost[depot][s] + cost[t][routex[-1][1]] - cost[depot][routex[-1][1]] < min_cost:
        min_cost = cost[depot][s] + cost[t][routex[-1][1]] - cost[depot][routex[-1][1]]
        min_pos = len(routex)
        min_rev = True
    if cost[depot][t] + cost[s][routex[-1][1]] - cost[depot][routex[-1][1]] < min_cost:
        min_cost = cost[depot][t] + cost[s][routex[-1][1]] - cost[depot][routex[-1][1]]
        min_pos = len(routex)
        min_rev = False
    for i in range(1, len(routex)):
        if cost[routex[i-1][1]][s] + cost[routex[i][0]][t] - cost[routex[i-1][1]][routex[i][0]] < min_cost:
            min_cost = cost[routex[i-1][1]][s] + cost[routex[i][0]][t] - cost[routex[i-1][1]][routex[i][0]]
            min_pos = i
            min_rev = False
        if cost[routex[i][0]][s] + cost[routex[i-1][1]][t] - cost[routex[i][0]][routex[i-1][1]] < min_cost:
            min_cost = cost[routex[i][0]][s] + cost[routex[i-1][1]][t] - cost[routex[i][0]][routex[i-1][1]]
            min_pos = i
            min_rev = True
    if min_rev:
        new_route = routex[:min_pos] + rev_route + routex[min_pos:]
    else:
        new_route = routex[:min_pos] + routey + routex[min_pos:]

    return new_route, calc_route_cost(new_route, instance, cost)

def modify(routes, ccost, instance, cost, rng):
    x = rng.integers(len(routes))
    y = rng.integers(len(routes))
    if x > y:
        x, y = y, x
    if len (routes[x]) < 2:
        return routes, ccost
    l, r = rng.choice(len(routes[x]), size=2, replace=True)
    if l > r:
        l, r = r, l
    while r-l+1 == len(routes[x]):
        l, r = rng.choice(len(routes[x]), size=2, replace=True)
        if l > r:
            l, r = r, l
    if x == y:
        routex, new_cost = best_modify(routes[x][:l] + routes[x][r+1:], routes[x][l:r+1], instance, cost)
        # if new_cost < calc_route_cost(routes[x], instance, cost):
        #     print(new_cost - calc_route_cost(routes[x], instance, cost))
        return routes[:x] + [routex] + routes[x+1:], ccost + new_cost - calc_route_cost(routes[x], instance, cost)
    else:
        demand = instance['edge_demand']
        capacity = instance['capacity']
        loadx = sum(demand[(min(u, v), max(u, v))] for u, v in routes[x])
        loady = sum(demand[(min(u, v), max(u, v))] for u, v in routes[y])
        if loadx + loady <= capacity:
            routex, new_cost = best_modify(routes[x], routes[y], instance, cost)
            origin = calc_route_cost(routes[x], instance, cost) + calc_route_cost(routes[y], instance, cost)
            if new_cost < origin:
                return routes[:x] + [routex] + routes[x+1:y] + routes[y+1:], ccost + new_cost - origin
            else:
                return routes, ccost
        else:
            ll = rng.integers(len(routes[y]))
            best_routes, best_cost = routes, ccost
            wei0 = 0
            origin = calc_route_cost(routes[x], instance, cost) + calc_route_cost(routes[y], instance, cost)
            for i in range(l,r+1):
                wei0 += demand[(min(routes[x][i][0], routes[x][i][1]), max(routes[x][i][0], routes[x][i][1]))]
            wei = 0
            for i in range(ll,len(routes[y])):
                wei += demand[(min(routes[y][i][0], routes[y][i][1]), max(routes[y][i][0], routes[y][i][1]))]
                if loadx - wei0 + wei > capacity:
                    break
                if loady - wei + wei0 > capacity:
                    continue
                new_routex, new_costx = best_modify(routes[x][:l] + routes[x][r+1:], routes[y][ll:i+1], instance, cost)
                new_routey, new_costy = best_modify(routes[y][:ll] + routes[y][i+1:], routes[x][l:r+1], instance, cost)
                if ccost + new_costx + new_costy - origin < best_cost:
                    best_cost = ccost + new_costx + new_costy - origin
                    best_routes = routes[:x] + [new_routex] + routes[x+1:y] + [new_routey] + routes[y+1:]
            return best_routes, best_cost

def main():
    instance_file, termination, seed = parse_args()
    rng = np.random.default_rng(seed)
    instance = parse_instance(instance_file)
    cost = build_graph(instance)
    start = time.time()
    best_routes = path_scanning(instance, cost, rng)
    best_cost = calc_total_cost(best_routes, instance, cost)

    max_iter = instance['V'] * instance['V'] << 2

    while time.time() - start < termination - 1:
        curr_routes = path_scanning(instance, cost, rng)
        curr_cost = calc_total_cost(curr_routes, instance, cost)
        # print('?')
        for i in range(max_iter):
            curr_routes, curr_cost = modify(curr_routes, curr_cost, instance, cost, rng)
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_routes = curr_routes
            if time.time() - start > termination - 1:
                break
    print(format_solution(best_routes))
    print(f'q {best_cost}')

if __name__ == '__main__':
    main()