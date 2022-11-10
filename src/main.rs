extern crate plotly;
use plotly::common::Mode;
use plotly::{Image, ImageFormat, Layout, Plot, Scatter};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::RngCore;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::Write;

#[derive(Debug)]
struct Arguments {
    in_file: String,
    p0: usize,
    cross_chance: f32,
    mut_chance: f32,
    shut_percent: f32,
    reopen_percent: f32,
    per_lock: f32,
}

fn parse_args() -> Arguments {
    let args: Vec<String> = env::args().skip(1).collect();

    Arguments {
        in_file: args[0].clone(),
        p0: args[1].clone().parse().unwrap(),
        cross_chance: args[2].parse().unwrap(),
        mut_chance: args[3].parse().unwrap(),
        shut_percent: args[4].parse().unwrap(),
        reopen_percent: args[5].parse().unwrap(),
        per_lock: args[6].parse().unwrap(),
    }
}
fn parse_graph(f: File) -> (Vec<(u32, u32)>, Vec<Vec<u32>>) {
    let mut reader = BufReader::new(f);
    let mut first_line: String = "".to_string();
    let _i = reader.read_line(&mut first_line);
    first_line = first_line.strip_suffix('\n').unwrap().to_owned();
    let split_first = first_line
        .split(' ')
        .map(|x| x.parse::<u32>().unwrap())
        .collect::<Vec<_>>();
    println!("{:?}", split_first);
    println!("{}", &first_line);
    let mut adj_list: Vec<Vec<u32>> = Vec::<Vec<u32>>::with_capacity(split_first[0] as usize);
    let mut edg_list: Vec<(u32, u32)> = vec![];
    for line_ in reader.lines().enumerate() {
        let line = line_.1.unwrap();
        let line: Vec<u32> = line.split(' ').map(|x| x.parse::<u32>().unwrap()).collect();

        let idx = line_.0;
        if !line.is_empty() {
            adj_list.push(vec![]);
            for numb in &line {
                adj_list[idx].push(*numb);
                if *numb > idx as u32 {
                    edg_list.push((idx as u32, *numb));
                }
            }
        }
    }
    (edg_list, adj_list)
}

#[derive(Clone, Debug)]
pub struct Individual {
    chromosome: Vec<usize>,
    fitness: f32,
    // lockdown_edgelist: Vec<(u32, u32)>,
    lockdown_adjlist: Vec<Vec<u32>>,
    adjlist: Vec<Vec<u32>>,
    mut_chance: f32,
}

impl Individual {
    fn mutate(&mut self, rng: &mut dyn RngCore, chance_mut: f32) {
        if rng.gen::<f32>() < chance_mut {
            let mut count = 0;
            while count < 2 {
                let idx = rng.gen_range(0..self.chromosome.len());
                let idx2 = rng.gen_range(0..self.chromosome.len());
                if self.chromosome[idx] == 0 && self.chromosome[idx2] == 1 {
                    self.chromosome[idx] = 1;
                    self.chromosome[idx2] = 0;
                    count += 1;
                } else if self.chromosome[idx] == 1 && self.chromosome[idx2] == 0 {
                    self.chromosome[idx] = 0;
                    self.chromosome[idx2] = 1;
                    count += 1;
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Population {
    pop: Vec<Individual>,
    cross_chance: f32,
    edg_list: Vec<(u32, u32)>,
    adj_list: Vec<Vec<u32>>,
    elitey: Individual,
    p0: usize,
}

impl Population {
    fn init_pop(
        percent_lock: f32,
        pop_size: i32,
        edg_list: &Vec<(u32, u32)>,
        adj_list: &[Vec<u32>],
        cross_chance: f32,
        mut_chance: f32,
        p0: usize,
    ) -> Self {
        let edg_count = edg_list.len();
        let mut pop_vec: Vec<Individual> = Vec::with_capacity(pop_size as usize);
        let num_remove = (percent_lock * edg_count as f32) as usize;
        // let mod_rem = num_remove % 2;
        // let num_remove = num_remove - mod_rem;
        let mut init_list = vec![0; edg_count];
        for (i, v) in init_list.iter_mut().enumerate() {
            *v = i as u32;
        }
        let mut rng = &mut rand::thread_rng();
        for _ in 0..pop_size {
            let mut new_list: Vec<usize> = vec![1; edg_count];
            let rindices: Vec<u32> = init_list
                .choose_multiple(&mut rng, num_remove)
                .cloned()
                .collect();
            for j in &rindices {
                new_list[*j as usize] = 0;
            }
            let (ladjlist, _ledgelist) = get_lockdown_graphs(adj_list, edg_list, &new_list);
            let new_ind = Individual {
                chromosome: new_list,
                fitness: 999999.0,
                // lockdown_edgelist: ledgelist,
                lockdown_adjlist: ladjlist,
                adjlist: adj_list.to_owned(),
                mut_chance: mut_chance,
            };
            pop_vec.push(new_ind);
        }
        Population {
            pop: pop_vec.clone(),
            cross_chance: cross_chance,
            edg_list: edg_list.clone(),
            adj_list: adj_list.to_owned(),
            elitey: pop_vec[0].clone(),
            p0: p0,
        }
    }

    fn tournament(&self, rng: &mut dyn RngCore, k: usize) -> Individual {
        let indices: Vec<_> = self.pop.choose_multiple(rng, k).collect();
        let mut best = indices[0];
        for i in &indices {
            if i.fitness < best.fitness {
                best = i;
            }
        }
        best.clone()
    }

    fn sdb(
        ind1: &Individual,
        ind2: &Individual,
        cross_chance: f32,
        rng: &mut dyn RngCore,
        edg_list: &[(u32, u32)],
        adj_list: &[Vec<u32>],
    ) -> (Individual, Individual) {
        let chance = rng.gen::<f32>();
        if chance < cross_chance {
            let mut ind1_set: HashSet<usize> = HashSet::new();
            let mut ind2_set: HashSet<usize> = HashSet::new();
            let mut set_intersect = HashSet::new();
            let mut intersect = vec![];
            let mut ind1_ones: Vec<usize> = vec![];
            let mut ind2_ones: Vec<usize> = vec![];
            for (idx, value) in ind1.chromosome.iter().enumerate() {
                if *value == 1 && ind2.chromosome[idx] == 1 {
                    intersect.push(idx);
                    set_intersect.insert(idx);
                } else if *value == 1 {
                    ind1_ones.push(idx);
                    ind1_set.insert(idx);
                }
            }
            for (idx, value) in ind2.chromosome.iter().enumerate() {
                if *value == 1 && !intersect.contains(&idx) {
                    ind2_ones.push(idx);
                    ind2_set.insert(idx);
                }
            }
            let extra_ones: HashSet<_> = ind1_set.union(&ind2_set).collect();
            let mut child1 = HashSet::new();
            let mut child2 = HashSet::new();
            let mut switch = false;
            for i in extra_ones {
                if switch {
                    child1.insert(*i);
                    switch = false;
                } else {
                    child2.insert(*i);
                    switch = true;
                }
            }
            child1.extend(&set_intersect);
            child2.extend(&set_intersect);
            let c1_tmp = Vec::from_iter(child1);
            let c2_tmp = Vec::from_iter(child2);
            let mut c1_vec = vec![0; ind1.chromosome.len()];
            let mut c2_vec = vec![0; ind2.chromosome.len()];
            for i in c1_tmp {
                c1_vec[i] = 1;
            }
            for i in c2_tmp {
                c2_vec[i] = 1;
            }
            let (c1_adj, _c1_edg) = get_lockdown_graphs(adj_list, edg_list, &c1_vec);
            let (c2_adj, _c2_edg) = get_lockdown_graphs(adj_list, edg_list, &c2_vec);
            let c1 = Individual {
                chromosome: c1_vec,
                fitness: 999999.0,
                // lockdown_edgelist: c1_edg,
                lockdown_adjlist: c1_adj,
                adjlist: adj_list.to_owned(),
                mut_chance: ind1.mut_chance,
            };
            let c2 = Individual {
                chromosome: c2_vec,
                fitness: 999999.0,
                // lockdown_edgelist: c2_edg,
                lockdown_adjlist: c2_adj,
                adjlist: adj_list.to_owned(),
                mut_chance: ind1.mut_chance,
            };
            return (c1, c2);
        }
        return (ind1.clone(), ind2.clone());
    }

    fn evolve(
        &mut self,
        max_gen: u32,
        p0: usize,
        alpha: f64,
        node_num: usize,
        shutdown_percent: f32,
        reopen_percent: f32,
        k: usize,
        rng: &mut dyn RngCore,
        count: usize,
    ) {
        let file_name = format!("results{}.txt", count);
        let mut file = File::create(file_name).expect("create failed");
        for gen in 0..=max_gen {
            let mut scores = vec![];
            let mut elite: Individual = self.pop[0].clone();
            for i in 0..self.pop.len() {
                let mut tmp = 0;
                for _ in 0..10 {
                    let mut fit = 9999999;
                    while fit >= 999999 {
                        let (_, _, score, _, _, _) = fitness_sirs(
                            &self.pop[i],
                            p0,
                            alpha,
                            node_num,
                            shutdown_percent,
                            reopen_percent,
                        );
                        if score >= 5 {
                            tmp += score;
                            fit = score;
                        }
                    }
                }
                let fitness: f32 = tmp as f32 / 10.0;
                scores.push(fitness);
                self.pop[i].fitness = fitness;
                if fitness < elite.fitness {
                    elite = self.pop[i].clone();
                }
            }
            let meany: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
            println!("Gen: {gen}, Best: {}, Mean: {}", elite.fitness, meany);
            let result = format!("Gen: {gen}, Best: {}, Mean: {}", elite.fitness, meany);
            writeln!(file, "{}", result).expect("write failed");
            let mut children: Vec<Individual> = vec![];
            children.push(elite.clone());
            self.elitey = elite;
            while children.len() < self.pop.len() {
                let p1 = self.tournament(rng, k);
                let p2 = self.tournament(rng, k);
                let (mut c1, mut c2) = Population::sdb(
                    &p1,
                    &p2,
                    self.cross_chance,
                    rng,
                    &self.edg_list,
                    &self.adj_list,
                );
                c1.mutate(rng, c1.mut_chance);
                c2.mutate(rng, c2.mut_chance);
                children.push(c1);
                children.push(c2);
            }
            self.pop = children;
        }
    }

    fn run_epis(&self, shut_per: f32, re_per: f32, count: usize) {
        let epi_total = 50;
        let mut epi_logs: Vec<Vec<usize>> = vec![];
        let mut lengths: Vec<usize> = vec![];
        let mut sums = vec![0; self.elitey.adjlist.len()];
        let mut counts = vec![0; self.elitey.adjlist.len()];
        let mut totals: Vec<u32> = vec![];
        let mut stops: Vec<u32> = vec![];
        let mut reopens: Vec<u32> = vec![];
        for _ in 0..epi_total {
            let (max, len, ttl, lock_step, re_step, epi_log) = fitness_sirs(
                &self.elitey,
                self.p0,
                0.3,
                self.elitey.adjlist.len(),
                shut_per,
                re_per,
            );
            lengths.push(len as usize);
            let new_epi_log: Vec<usize> = epi_log.iter().map(|x| x.len()).collect();
            // println!("{:?}", &new_epi_log);
            epi_logs.push(new_epi_log);
            stops.push(lock_step);
            totals.push(ttl);
            reopens.push(re_step);
        }
        let avg_total: f32 = totals.iter().sum::<u32>() as f32 / totals.len() as f32;
        for (ln, el) in epi_logs.iter().enumerate() {
            for day in 0..(lengths[ln]) {
                sums[day] += el[day];
                counts[day] += 1;
            }
        }
        let mut avg = vec![];
        let mut avg_all: Vec<f32> = vec![];
        for (day, s) in sums.iter().enumerate() {
            if counts[day] > 0 {
                avg.push(s / counts[day]);
                avg_all.push((s / epi_total) as f32);
            }
        }
        avg.push(0);
        avg_all.push(0.0);
        let max_len = *lengths.iter().max().unwrap();
        // for mut el in &mut epi_logs {
        //     let s_len = el.len();
        //     for _ in 0..(max_len - s_len) {
        //         el.push(0);
        //     }
        // }
        let x: Vec<usize> = (0..=max_len).collect();
        let mut x_lbls: Vec<_> = x.clone().iter().map(|x| x.to_string()).collect();
        let trace1 = Scatter::new(x.clone(), avg_all)
            .name("Avg_All")
            .mode(Mode::Lines);
        let trace2 = Scatter::new(x.clone(), avg)
            .name("Avg_Running")
            .mode(Mode::Lines);
        let mut plot = Plot::new();
        plot.add_trace(trace1);
        plot.add_trace(trace2);
        plot.show_image(ImageFormat::PNG, 1280, 900);
        let plot_name = format!("epi_plot{}.png", count);
        plot.write_image(plot_name, ImageFormat::PNG, 1280, 900, 1.0);

        // python! {
        //     import matplotlib.pyplot as plt
        //     import matplotlib.pyplot
        //     print("testy1")
        //     fig = matplotlib.pyplot.figure()
        //     print("testy2")
        //     fig.set_dpi(400)
        //     fig.set_figheight(4)
        //     plot = fig.add_subplot(111)
        //     // for el in 'epi_logs:
        //     //     if len(el) > 5:
        //     //         plot.plot('x, el, linewidth=1, alpha=0.3, color="gray")
        //     print("Test1")
        //     plot.plot('x, 'avg, label="Average of Running")
        //     plot.plot('x, 'avg_all, label="Average of All")
        //     print("Test2")
        //     fig.suptitle("Epidemic Profiles for 50 Epidemics")
        //     plot.set_ylabel("Newly Infected Individuals")
        //     plot.set_xlabel("Day \n DATA: Avg Infected: " + str('avg_total))
        //     plot.set_xticks('x)
        //     plot.set_xticklabels('x_lbls)
        //     plot.legend()
        //     fig.tight_layout()
        //     // plot.imshow()
        //     fig.savefig("test_epi_fig.png")

        // }
    }
}

fn get_lockdown_graphs(
    adj_list: &[Vec<u32>],
    edg_list: &[(u32, u32)],
    remove_list: &[usize],
) -> (Vec<Vec<u32>>, Vec<(u32, u32)>) {
    let mut lock_adj: Vec<Vec<u32>> = adj_list.to_owned();
    let mut lock_edg: Vec<(u32, u32)> = edg_list.to_owned();
    let mut removeylist: Vec<usize> = vec![];
    for (idx, edge) in remove_list.iter().enumerate() {
        if *edge == 0 {
            let first = edg_list[idx].0;
            let second = edg_list[idx].1;
            if let Some(pos) = lock_adj[first as usize].iter().position(|x| *x == second) {
                lock_adj[first as usize].remove(pos);
            }
            if let Some(pos) = lock_adj[second as usize].iter().position(|x| *x == first) {
                lock_adj[second as usize].remove(pos);
            }
            removeylist.push(idx);
        }
    }
    for x in removeylist.iter().rev() {
        lock_edg.remove(*x);
    }
    (lock_adj, lock_edg)
}

pub fn set_colour(value: u32, node_num: usize) -> Vec<u32> {
    let clr = vec![value; node_num];
    clr
}

pub fn infected(n: u32, alpha: f64) -> u32 {
    let mut rng = rand::thread_rng();
    let beta: f64 = 1.0 - (f64::from(n) * (1.0 - alpha).ln()).exp();
    let x = rng.gen::<f64>();
    if x < beta {
        return 1;
    } else {
        return 0;
    }
}

pub fn fitness_sirs(
    individual: &Individual,
    p0: usize,
    alpha: f64,
    node_num: usize,
    shutdown_percent: f32,
    reopen_percent: f32,
) -> (u32, u32, u32, u32, u32, Vec<Vec<usize>>) {
    if p0 >= node_num {
        return (0, 0, 0, 0, 0, vec![vec![0]]);
    }
    let mut max = 0;
    let mut len = 0;
    let mut ttl = 0;
    let mut have_locked_down = false;
    let mut have_reopened = false;
    let mut reopen_step = 9999;
    let mut time_step = 0;
    let mut lockdown_step = 0;
    let mut epi_log: Vec<Vec<usize>> = Vec::new();
    epi_log.push(vec![p0]);
    let mut nin: Vec<u32> = vec![0; node_num]; //infected neibours counters
    let mut clr: Vec<u32> = vec![0; node_num]; // set population to susceptible
    clr[p0] = 1; //infect patient zero
    let mut numb_inf = 1; // initialize to one person currently infected
    let mut tmp_list = &individual.adjlist;
    while numb_inf > 0 && time_step < node_num as u32 {
        let current_infected = numb_inf as f32 / node_num as f32;
        if (current_infected >= shutdown_percent) && !have_locked_down {
            tmp_list = &individual.lockdown_adjlist;
            have_locked_down = true;
            lockdown_step = time_step;
        }
        for i in nin.iter_mut().take(node_num) {
            *i = 0; //zero the number of infected neighbors buffer
        }
        //if threshold met then restore initial contact graph
        // current_infected = numb_inf as f32 / node_num as f32;

        if (current_infected < reopen_percent) && have_locked_down && !have_reopened {
            tmp_list = &individual.adjlist;
            reopen_step = time_step;
            have_reopened = true;
        }

        for i in 0..node_num {
            if clr[i] == 1 {
                //found infected individual
                for j in 0..tmp_list[i].len() {
                    nin[tmp_list[i][j] as usize] += 1; //record exposure
                }
            }
        }
        //check for transmission
        for i in 0..node_num {
            if clr[i] == 0 && nin[i] > 0 {
                clr[i] = 3 * infected(nin[i], alpha);
            }
        }
        if numb_inf > max {
            max = numb_inf;
        }
        ttl += numb_inf;
        numb_inf = 0;
        let mut curr_epi = vec![];
        for i in 0..node_num {
            match clr[i] {
                0 => (),         //susceptible, do nothing
                1 => clr[i] = 2, //infected, move to removed
                // 2 => (),         //removed, move to susceptible
                2 => clr[i] = 4, //removed, move to removed2
                3 => {
                    //newly infected
                    clr[i] = 1;
                    numb_inf += 1;
                    curr_epi.push(i);
                }
                4 => clr[i] = 5, //removed, move to removed3
                5 => clr[i] = 0, //removed, move to susceptible
                _ => (),
            }
        }
        epi_log.push(curr_epi);
        time_step += 1;
        len += 1; //record time step
    }
    (max, len, ttl, lockdown_step, reopen_step, epi_log)
}

fn main() -> std::io::Result<()> {
    static ALPHA: f64 = 0.3;
    let mut rng = rand::thread_rng();
    let args = parse_args();
    println!("{}", args.p0);
    let f = File::open(&args.in_file).unwrap();

    let (elist, alist) = parse_graph(f);
    for i in 0..30 {
        let mut pop = Population::init_pop(
            args.per_lock,
            201,
            &elist,
            &alist,
            args.cross_chance,
            args.mut_chance,
            args.p0,
        );
        pop.evolve(
            100,
            args.p0,
            ALPHA,
            alist.len(),
            args.shut_percent,
            args.reopen_percent,
            7,
            &mut rng,
            i,
        );
        pop.run_epis(args.shut_percent, args.reopen_percent, i);
        let (elite_adj, elite_edj) =
            get_lockdown_graphs(&pop.adj_list, &pop.edg_list, &pop.elitey.chromosome);

        let lgraph_name = format!("lockdown_graph{}.dat", i);
        let mut output = File::create(lgraph_name).expect("create failed");
        let header = format!("Nodes: {}, Edges: {}", alist.len(), elite_edj.len());
        writeln!(output, "{}", header).expect("write failed");
        for node in &elite_adj {
            for n in node.iter() {
                write!(output, "{} ", n)?;
            }
            writeln!(output, "")?;
        }
    }

    // let specs = GraphSpecs::multi_undirected();
    // let mut nodestmp: Vec<_> = (0..alist.len()).collect();
    // let mut nodes: Vec<Node<String, ()>> = vec![];
    // for i in &nodestmp {
    //     nodes.push(Node::from_name(i.to_string()));
    // }
    // let mut edges: Vec<Edge<String, ()>> = vec![];
    // for i in &elite_edj {
    //     edges.push(Edge::with_weight(i.0.to_string(), i.1.to_string(), 1.0));
    // }
    // let mut graph: Graph<String, ()> = Graph::new(specs);
    // graph.add_nodes(nodes);
    // // graph.add_edges(edges);
    // for i in &elite_edj {
    //     graph.add_edge_tuple(i.0.to_string(), i.1.to_string());
    // }
    // let new_graph = graph.to_single_edges().unwrap();

    println!("cat");
    Ok(())
    // println!("{:?}", pop.pop[4]);
    // println!("{:?}", alist);
    // println!("{:?}", elist);
}
