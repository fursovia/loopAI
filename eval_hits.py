from projects.convai2.eval_hits import setup_args, eval_hits

if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='projects.loopAI.t_agent:DSSMAgent',
        rank_candidates=True,
    )
    opt = parser.parse_args(print_args=False)
    eval_hits(opt, print_parser=parser)
