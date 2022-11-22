from graphviz import Graph
import sys
import os

outp = "./Output/"
lower_better = True


def high_low_deg(el: list, verts: int):
    deg = [int(0) for _ in range(verts)]
    low_deg = []
    high_deg = []
    for ed in el:
        deg[ed[0]] += 1
        deg[ed[1]] += 1
        pass
    max = 0
    for idx, deg in enumerate(deg):
        if deg > max:
            max = deg
            pass
        if deg > 20:
            high_deg.append(idx)
            pass
        elif deg > 10:
            low_deg.append(idx)
            pass
        pass
    print("Max: " + str(max))
    return low_deg, high_deg


def edge_list(filename):
    el = []
    with open(filename) as f:
        lines = f.readlines()
        lines.__delitem__(0)
        for from_node, line in enumerate(lines):
            line = line.rstrip()
            line = line.split(" ")
            for to_node in line:
                if to_node != '':
                    # if [from_node, int(to_node)] not in el:
                    # if [int(to_node), from_node] not in el:
                    el.append([from_node, int(to_node)])
                    # pass
                    # pass
                    pass
                pass
            pass
        pass
    edge_lists = []
    edge_counts = []
    for d in el:
        if d not in edge_lists:
            edge_lists.append(d)
            edge_counts.append(el.count(d))
            pass
        pass

    return edge_lists, edge_counts


def make_graph(el: list, ec: list, low_deg: list, high_deg: list, out_file: str, verts: int, p0: int):
    g = Graph(engine='sfdp')
    e_cout = 0

    g.graph_attr.update(dpi='1000', size="10,10", outputorder='edgesfirst', overlap='false', splines='true')
    g.node_attr.update(color='black', shape='point', width='0.02', height='0.02')
    g.edge_attr.update(color='black', penwidth='0.5')
    for n in range(verts):
        if n == p0:
            if n in low_deg:
                g.node(str(n), label=str(n), color='red', width='0.03', height='0.03')
                pass
            elif n in high_deg:
                g.node(str(n), label=str(n), color='red', width='0.04', height='0.04')
                pass
            else:
                g.node(str(n), label=str(n), color='red')
                pass
        elif n in low_deg:
            g.node(str(n), label=str(n), width='0.03', height='0.03')
        elif n in high_deg:
            g.node(str(n), label=str(n), width='0.04', height='0.04')
        else:
            g.node(str(n), label=str(n))
        pass

    ew_count = 0
    counts = [0 for _ in range(5)]
    for idx, d in enumerate(el):
        if d[0] < d[1]:
            if ec[idx] == 1:
                g.edge(str(d[0]), str(d[1]), color='black')
                pass
            elif ec[idx] == 2:
                g.edge(str(d[0]), str(d[1]), color='purple')
                pass
            elif ec[idx] == 3:
                g.edge(str(d[0]), str(d[1]), color='blue')
                pass
            elif ec[idx] == 4:
                g.edge(str(d[0]), str(d[1]), color='orange')
                pass
            else:
                g.edge(str(d[0]), str(d[1]), color='red')
                pass

            # g.edge(str(d[0]), str(d[1]), penwidth=str(pw * ec[idx]))
            e_cout += 1
            ew_count += ec[idx]
            counts[ec[idx] - 1] += 1
            pass
        pass

    print("Edges: " + str(e_cout))
    print("Total Weights: " + str(ew_count))
    print(counts)
    g.render(filename=out_file, directory=outp, cleanup=True, format='png')
    g.save(filename=out_file, directory=outp)
    pass


def main():
    # nodes = [256, 512, 768, 1024]
    # nodes = [256]
    smallP0 = [212, 132, 118]
    largeP0 = [479, 43, 479]
    for dirpath, dirnames, files in os.walk('.'):
        for file_name in files:
            if file_name.__contains__("lockdown_graph") and file_name.endswith(".dat"):
                exper = dirpath.split('/')
                exper = exper[-1]
                fi = file_name.split('.')
                fi = exper + fi[0]
                direc = os.path.join(dirpath, file_name)
                el, ec = edge_list(direc)
                low_deg, high_deg = high_low_deg(el, 512)
                make_graph(el, ec, low_deg, high_deg, fi, 512)
                print("done 1")
                pass

                print(dirpath)
    # finames = ["lockdown_graph13"]
    # for fi in finames:
    #     el, ec = edge_list(fi + ".dat")
    #     low_deg, high_deg = high_low_deg(el, 256)
    #     make_graph(el, ec, low_deg, high_deg, fi, 256)
    #     print("done 1")
    #     pass
    print("DONE")
    pass


main()
