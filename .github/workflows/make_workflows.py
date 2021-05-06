#!/usr/bin/env python3

import jinja2
import yaml

if __name__ == '__main__':
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'),
                             block_start_string = '<%',
                             block_end_string = '%>',
                             variable_start_string = '<<',
                             variable_end_string = '>>',
                             comment_start_string = '<#',
                             comment_end_string = '#>',
                             trim_blocks=True,
                             lstrip_blocks=True)
    template = env.get_template('test.yml')
    with open('templates/configurations.yml', 'r') as f:
        configurations = yaml.safe_load(f)

    # preprocess configurations and fill out additional fields needed by
    # `test.yml` to configure the matrix jobs
    for name,configuration in configurations.items():
        for entry in configuration:
            if entry['config'].startswith('[cuda'):
                entry['runner'] = "[self-hosted,GPU]"
                entry['docker_options'] = "--gpus=all"
            else:
                entry['runner'] = "ubuntu-latest"
                entry['docker_options'] = ""

    with open('test.yml', 'w') as f:
        f.write(template.render(configurations))

    template = env.get_template('release.yml')
    with open('release.yml', 'w') as f:
        f.write(template.render())
