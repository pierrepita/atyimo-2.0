import sys
import yaml
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, IntegerType
import pyspark.sql.functions as F

# UDFs assumidas como disponíveis no mesmo diretório
from Functions import abf, clk, rlb

def carregar_configuracao(spark, yaml_path):
    if yaml_path.startswith("hdfs://"):
        temp_path = "/tmp/config.yaml"
        spark.sparkContext.addFile(yaml_path)
        os.system(f"hdfs dfs -copyToLocal {yaml_path} {temp_path}")
    else:
        temp_path = yaml_path

    with open(temp_path, 'r') as f:
        return yaml.safe_load(f)

def criar_udf(tipo, params):
    if tipo == 'abf':
        return F.udf(lambda x: abf(x, params['m'], params['k'], params['t_hash']), ArrayType(IntegerType()))
    elif tipo == 'clk':
        return F.udf(lambda x: clk(x, params['m'], params['k'], params['t_hash'], params['positions']), ArrayType(IntegerType()))
    elif tipo == 'rlb':
        return F.udf(lambda x: rlb(x, params['m'], params['k'], params['t_hash'], params['positions'], params['vetor'], params['a']), ArrayType(IntegerType()))
    else:
        raise ValueError(f"Tipo de Bloom Filter inválido: {tipo}")

def carregar_dados(spark, caminho):
    if caminho.endswith(".csv"):
        return spark.read.option("header", True).csv(caminho)
    elif caminho.endswith(".parquet"):
        return spark.read.parquet(caminho)
    else:
        raise ValueError("Formato de arquivo não suportado.")

def processar(df, config, spark):
    vars_codificar = config['variaveis_codificar']
    tipo = config['tipo_bloom']
    params = config['parametros']

    for v in vars_codificar:
        if v not in df.columns:
            raise ValueError(f"Variável '{v}' não encontrada no DataFrame.")

    df = df.withColumn("vet", F.array(*[F.col(c) for c in vars_codificar]))
    bloom_udf = criar_udf(tipo, params)
    df = df.withColumn("bloom", bloom_udf(F.col("vet"))).drop("vet")

    return df

def main(caminho_config, caminho_a, caminho_b):
    spark = SparkSession.builder.appName("BloomFilterGeneration").getOrCreate()

    config = carregar_configuracao(spark, caminho_config)
    caminho_saida = config['saida_parquet']  # exemplo: hdfs://namenode:9000/output/bloom_result_a

    df_a = carregar_dados(spark, caminho_a)
    df_b = carregar_dados(spark, caminho_b)

    df_a_result = processar(df_a, config, spark)
    df_b_result = processar(df_b, config, spark)

    df_a_result.write.mode("overwrite").parquet(caminho_saida + "/base_a")
    df_b_result.write.mode("overwrite").parquet(caminho_saida + "/base_b")

    print(f"Dados escritos em: {caminho_saida}/base_a e {caminho_saida}/base_b")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: spark-submit gerar_bloom.py <config.yaml> <base_a> <base_b>")
        sys.exit(1)

    caminho_config = sys.argv[1]
    caminho_a = sys.argv[2]
    caminho_b = sys.argv[3]
    main(caminho_config, caminho_a, caminho_b)
