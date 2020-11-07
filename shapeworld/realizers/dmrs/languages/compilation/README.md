# Custom compilation of the ERG grammar


The ERG grammar files can be obtained via SVN:

```bash
svn checkout http://svn.delph-in.net/erg/trunk erg
```

Subsequently, the following script creates a copy of the grammar files, applies the changes necessary for ShapeWorld, and compiles the grammar using the framework version of ACE:

```bash
./compile.sh erg erg-shapeworld
```

The `erg` folder is not required anymore and can be removed.
