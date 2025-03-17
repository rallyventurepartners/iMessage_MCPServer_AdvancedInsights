from setuptools import setup, find_packages

setup(
    name="imessage_advanced_insights",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "Flask>=2.2.3",
        "networkx>=2.8.8",
        "numpy>=1.24.2",
        "python-louvain>=0.16",
        "textblob>=0.17.1",
        "phonenumbers>=8.13.5",
        "python-dateutil>=2.8.2",
        "nltk>=3.8.1",
        "scikit-learn>=1.2.2",
        "spacy>=3.5.1",
        "wordcloud>=1.8.2.2",
        "matplotlib>=3.7.1",
        "pandas>=2.0.0",
        "jsonschema>=4.17.3",
        "requests>=2.28.2"
    ],
    entry_points={
        "console_scripts": [
            "imessage-insights=main:main",
        ],
    },
    author="David Jelinek",
    author_email="david@rvp.io",
    description="A powerful analysis server for extracting insights from iMessage conversations",
    keywords="imessage, analytics, sentiment, network, visualization",
    url="https://github.com/rallyventurepartners/iMessage_MCPServer_AdvancedInsights",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Communications :: Chat",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
) 